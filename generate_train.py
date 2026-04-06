import os
import re
import pickle
import warnings
import gc
import argparse
from collections import Counter

import numpy as np
import torch
from torch_geometric.data import Data
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from Bio import SeqIO

try:
    from Bio.PDB.Polypeptide import three_to_one
except ImportError:
    from Bio.SeqUtils import seq1 as three_to_one

from tqdm import tqdm
import esm


def parse_args():
    parser = argparse.ArgumentParser(description="GeoMEL Training Data Generator")
    parser.add_argument("--train_fasta", type=str, required=True)
    parser.add_argument("--train_pdb_dirs", type=str, nargs="+", required=True)
    parser.add_argument("--output_dir", type=str, default="./data2/processed")
    parser.add_argument("--esm_model", type=str, default="esm2_t36_3B_UR50D")
    parser.add_argument("--esm_dim", type=int, default=2560)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--esm_batch_tokens", type=int, default=2048)
    parser.add_argument("--steps", type=str, nargs="+", default=["labels", "graphs", "esm", "verify"],
                        choices=["labels", "graphs", "esm", "verify", "all"])
    return parser.parse_args()


LABEL_PATTERNS = [
    (r'T1SE|T1SS|Type.?1|type.?1', 1),
    (r'T2SE|T2SS|Type.?2|type.?2', 2),
    (r'T3SE|T3SS|Type.?3|type.?3', 3),
    (r'T4SE|T4SS|Type.?4|type.?4', 4),
    (r'T6SE|T6SS|Type.?6|type.?6', 5),
    (r'Non.?[Ee]ffector|non.?effector|negative|NEG', 0),
]

CLASS_NAMES = ['Non-Effector', 'T1SE', 'T2SE', 'T3SE', 'T4SE', 'T6SE']
AA_IDX = {a: i for i, a in enumerate('ACDEFGHIKLMNPQRSTVWYX')}


def extract_label(header):
    for pattern, idx in LABEL_PATTERNS:
        if re.search(pattern, header, re.IGNORECASE):
            return idx
    return None


def generate_labels(train_fasta, label_file):
    train_labels = {}
    train_sequences = {}
    for record in SeqIO.parse(train_fasta, 'fasta'):
        lbl = extract_label(record.description)
        if lbl is not None:
            train_labels[record.id] = lbl
            train_sequences[record.id] = str(record.seq)
    os.makedirs(os.path.dirname(label_file), exist_ok=True)
    with open(label_file, 'wb') as f:
        pickle.dump(train_labels, f)
    cnt = Counter(train_labels.values())
    print(f"train_labels.pkl generated: {len(train_labels)} sequences")
    print(f"saved to: {label_file}")
    for i in range(len(CLASS_NAMES)):
        print(f"  {CLASS_NAMES[i]:<16} (label={i}): {cnt.get(i, 0):>5}")
    with open(label_file, 'rb') as f:
        verify = pickle.load(f)
    assert verify == train_labels
    print("verification passed")
    return train_labels, train_sequences


def pdb_to_graph(pdb_path):
    parser = PDBParser(QUIET=True)
    try:
        struct = parser.get_structure('p', pdb_path)
    except:
        return None
    residues = [r for m in struct for c in m for r in c
                if is_aa(r, standard=True) and 'CA' in r]
    if len(residues) < 5:
        return None
    node_s, node_v, ca_coords = [], [], []
    for res in residues:
        try:
            n = res['N'].get_coord()
            ca = res['CA'].get_coord()
            c = res['C'].get_coord()
            ca_coords.append(ca)
            v1, v2 = n - ca, c - ca
            cos_angle = np.clip(
                np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8),
                -1, 1
            )
            angle = np.arccos(cos_angle)
            aa_idx = AA_IDX.get(three_to_one(res.get_resname()), 20)
            node_s.append([
                np.linalg.norm(c - n) / 10,
                np.linalg.norm(ca - n) / 10,
                np.linalg.norm(c - ca) / 10,
                angle / np.pi,
                np.sin(angle),
                aa_idx / 20
            ])
            u1 = (c - ca) / (np.linalg.norm(c - ca) + 1e-8)
            u2 = np.cross(n - ca, c - ca)
            u2 /= (np.linalg.norm(u2) + 1e-8)
            node_v.append([u1, u2, np.cross(u1, u2)])
        except:
            node_s.append([0] * 6)
            node_v.append([[0, 0, 0]] * 3)
            ca_coords.append([0, 0, 0])
    ca_coords = np.array(ca_coords)
    N = len(ca_coords)
    diff = ca_coords[:, None] - ca_coords[None, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    mask = (dist < 10) & (dist > 0)
    for i in range(N - 1):
        mask[i, i + 1] = mask[i + 1, i] = True
    src, dst = np.where(mask)
    if len(src) == 0:
        return None
    d = dist[src, dst]
    edge_s = np.exp(-((d[:, None] - np.linspace(0, 20, 32)) ** 2) / 2).astype(np.float32)
    direction = diff[src, dst]
    direction /= (np.linalg.norm(direction, axis=-1, keepdims=True) + 1e-8)
    return Data(
        node_s=torch.tensor(node_s, dtype=torch.float),
        node_v=torch.tensor(node_v, dtype=torch.float),
        edge_index=torch.tensor([src, dst], dtype=torch.long),
        edge_s=torch.tensor(edge_s),
        edge_v=torch.tensor(direction[:, None, :].astype(np.float32)),
        num_nodes=N
    )


def build_graphs(pdb_dirs, target_ids, output_path):
    if os.path.exists(output_path):
        print(f"  {output_path} already exists, loading")
        return torch.load(output_path, weights_only=False)
    if isinstance(pdb_dirs, str):
        pdb_dirs = [pdb_dirs]
    pdb_files = {}
    for d in pdb_dirs:
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.endswith('.pdb'):
                    pid = os.path.splitext(f)[0]
                    if pid not in pdb_files:
                        pdb_files[pid] = os.path.join(d, f)
    print(f"  found {len(pdb_files)} PDB files")
    graphs = {}
    failed = []
    for pid in tqdm(target_ids, desc="building graphs"):
        for tp in [pid, pid.replace('|', '_').replace('~', '_').replace('/', '_')]:
            if tp in pdb_files:
                g = pdb_to_graph(pdb_files[tp])
                if g:
                    graphs[pid] = g
                break
        else:
            failed.append(pid)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(graphs, output_path)
    print(f"  success {len(graphs)}/{len(target_ids)}")
    if failed:
        print(f"  missing PDB IDs ({len(failed)}): {failed[:10]}{'...' if len(failed) > 10 else ''}")
    return graphs


def extract_esm_features(sequences_dict, model_name, output_path, batch_tokens, device):
    if os.path.exists(output_path):
        print(f"  {output_path} already exists, loading")
        with open(output_path, 'rb') as f:
            return pickle.load(f)
    print(f"  loading ESM-2 model: {model_name} ...")
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    batch_converter = alphabet.get_batch_converter()
    model = model.eval().to(device)
    num_layers = model.num_layers
    print(f"  layers: {num_layers}, embed_dim: {model.embed_dim}")
    data_list = [(pid, seq) for pid, seq in sequences_dict.items()]
    data_list.sort(key=lambda x: len(x[1]), reverse=True)
    esm_features = {}
    current_batch = []
    current_tokens = 0
    pbar = tqdm(total=len(data_list), desc="ESM-2 extraction")

    def process_batch(batch_data):
        if not batch_data:
            return
        labels_b, strs_b, tokens_b = batch_converter(batch_data)
        tokens_b = tokens_b.to(device)
        with torch.no_grad():
            results = model(tokens_b, repr_layers=[num_layers], return_contacts=False)
        representations = results["representations"][num_layers]
        for i, (pid, seq) in enumerate(batch_data):
            seq_len = len(seq)
            token_repr = representations[i, 1:seq_len + 1, :]
            esm_features[pid] = token_repr.cpu().numpy()
        pbar.update(len(batch_data))

    for pid, seq in data_list:
        seq_tokens = len(seq) + 2
        if seq_tokens > batch_tokens:
            process_batch(current_batch)
            current_batch = []
            current_tokens = 0
            process_batch([(pid, seq)])
            continue
        if current_tokens + seq_tokens > batch_tokens and current_batch:
            process_batch(current_batch)
            current_batch = []
            current_tokens = 0
        current_batch.append((pid, seq))
        current_tokens += seq_tokens
    process_batch(current_batch)
    pbar.close()
    del model
    torch.cuda.empty_cache()
    gc.collect()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(esm_features, f, protocol=4)
    return esm_features


def verify_all(label_file, graph_file, esm_file, esm_dim):
    print("=" * 60)
    print("GeoMEL Data Verification Report")
    print("=" * 60)
    with open(label_file, 'rb') as f:
        v_labels = pickle.load(f)
    v_graphs = torch.load(graph_file, weights_only=False)
    with open(esm_file, 'rb') as f:
        v_esm = pickle.load(f)
    print(f"\n1. files loaded:")
    print(f"   train_labels.pkl : {len(v_labels)}")
    print(f"   train_graphs.pt  : {len(v_graphs)}")
    print(f"   train_esm.pkl    : {len(v_esm)}")
    valid_ids = sorted(set(v_labels) & set(v_esm) & set(v_graphs))
    print(f"\n2. valid samples (intersection): {len(valid_ids)}")
    label_only = set(v_labels) - set(v_graphs) - set(v_esm)
    no_graph = set(v_labels) & set(v_esm) - set(v_graphs)
    no_esm = set(v_labels) & set(v_graphs) - set(v_esm)
    if label_only:
        print(f"   labels only: {len(label_only)}")
    if no_graph:
        print(f"   missing graphs: {len(no_graph)}")
    if no_esm:
        print(f"   missing ESM: {len(no_esm)}")
    print(f"\n3. class distribution:")
    valid_cnt = Counter(v_labels[pid] for pid in valid_ids)
    for i in range(len(CLASS_NAMES)):
        n = valid_cnt.get(i, 0)
        pct = 100 * n / len(valid_ids) if valid_ids else 0
        print(f"   {CLASS_NAMES[i]:<16} (label={i}): {n:>5} ({pct:5.1f}%)")
    print(f"\n4. sample check:")
    if valid_ids:
        sample = valid_ids[0]
        g = v_graphs[sample]
        e = v_esm[sample]
        print(f"   ID: {sample}")
        print(f"   label: {v_labels[sample]} ({CLASS_NAMES[v_labels[sample]]})")
        print(f"   graph: nodes={g.num_nodes}, edges={g.edge_index.shape[1]}")
        print(f"   ESM: shape={e.shape}, dtype={e.dtype}")
    print(f"\n5. dimension check:")
    dim_ok = True
    mismatch_count = 0
    for pid in valid_ids:
        e_dim = v_esm[pid].shape[1]
        if e_dim != esm_dim:
            print(f"   FAIL {pid}: ESM dim {e_dim} != expected {esm_dim}")
            dim_ok = False
            break
        if v_graphs[pid].num_nodes != v_esm[pid].shape[0]:
            mismatch_count += 1
    if dim_ok:
        print(f"   ESM embedding dim: all {esm_dim} OK")
    if mismatch_count > 0:
        print(f"   graph nodes != ESM length: {mismatch_count} (auto-aligned during training)")
    else:
        print(f"   graph nodes == ESM length: all consistent")
    print(f"\n6. file sizes:")
    for fp, name in [(label_file, 'labels'), (graph_file, 'graphs'), (esm_file, 'esm')]:
        if os.path.exists(fp):
            sz = os.path.getsize(fp) / 1024 / 1024
            print(f"   {name:<8}: {sz:>8.1f} MB")
    print(f"\n{'=' * 60}")
    print(f"GeoMEL preprocessing complete.")
    print(f"{'=' * 60}")


def main():
    args = parse_args()
    warnings.filterwarnings("ignore")
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(args.gpu_id)}")

    label_file = f"{args.output_dir}/train_labels.pkl"
    graph_file = f"{args.output_dir}/train_graphs.pt"
    esm_file = f"{args.output_dir}/features_esm2_t36_3B/train_esm.pkl"

    steps = args.steps
    if "all" in steps:
        steps = ["labels", "graphs", "esm", "verify"]

    train_labels, train_sequences = None, None

    if "labels" in steps or "graphs" in steps or "esm" in steps:
        train_labels, train_sequences = generate_labels(args.train_fasta, label_file)

    if "graphs" in steps:
        train_ids = set(train_labels.keys())
        print(f"\nbuilding {len(train_ids)} protein structure graphs")
        build_graphs(args.train_pdb_dirs, train_ids, graph_file)

    if "esm" in steps:
        print(f"\nextracting ESM-2 features for {len(train_sequences)} sequences")
        extract_esm_features(train_sequences, args.esm_model, esm_file,
                             args.esm_batch_tokens, device)

    if "verify" in steps:
        verify_all(label_file, graph_file, esm_file, args.esm_dim)


if __name__ == "__main__":
    main()
