import re
import argparse
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from tqdm import tqdm

from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from Bio import SeqIO

try:
    from Bio.PDB.Polypeptide import three_to_one
except ImportError:
    from Bio.SeqUtils import seq1 as three_to_one

warnings.filterwarnings("ignore")

# =============================================================================
# Configuration
# =============================================================================

CLASS_NAMES = ['Non-Effector', 'T1SE', 'T2SE', 'T3SE', 'T4SE', 'T6SE']
NUM_CLASSES = 6

ESM_DIM = 2560
SEQ_HID = 256
STRUCT_HID = 100
CLS_D_MODEL = 128
CLS_N_HEADS = 4
CLS_N_LAYERS = 2

AA_IDX = {a: i for i, a in enumerate('ACDEFGHIKLMNPQRSTVWYX')}


# =============================================================================
# GVP model
# =============================================================================

class GVP(nn.Module):
    def __init__(self, in_dims, out_dims, act=(F.relu, torch.sigmoid)):
        super().__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.act_s, self.act_v = act
        self.Wh = nn.Linear(self.vi, self.vo, bias=False) if self.vi > 0 else None
        self.Ws = nn.Linear(self.si + (self.vo if self.vi > 0 else 0), self.so)

    def forward(self, x):
        s, v = x
        if self.Wh and v is not None:
            v = self.Wh(v.transpose(-1, -2)).transpose(-1, -2)
            s = torch.cat([s, torch.norm(v, dim=-1)], -1)
        s = self.Ws(s)
        if self.act_s:
            s = self.act_s(s)
        if self.vo > 0 and v is not None and self.act_v:
            v = v * self.act_v(torch.norm(v, dim=-1, keepdim=True))
        else:
            v = None
        return s, v


class GVPConv(nn.Module):
    def __init__(self, in_dims, out_dims, edge_dims):
        super().__init__()
        self.msg = GVP((2 * in_dims[0] + edge_dims[0], 2 * in_dims[1] + edge_dims[1]), out_dims)

    def forward(self, x, edge_index, edge_attr):
        s, v = x
        es, ev = edge_attr
        src, dst = edge_index
        ms = torch.cat([s[src], s[dst], es], -1)
        mv = torch.cat([v[src], v[dst], ev], 1) if v is not None else ev
        ms, mv = self.msg((ms, mv))
        from torch_scatter import scatter_mean
        return (scatter_mean(ms, dst, dim=0, dim_size=s.size(0)),
                scatter_mean(mv, dst, dim=0, dim_size=s.size(0)) if mv is not None else None)


class GVPEncoder(nn.Module):
    def __init__(self, node_dims=(6, 3), edge_dims=(32, 1), hidden=(100, 16), layers=3):
        super().__init__()
        self.node_emb = GVP(node_dims, hidden)
        self.edge_emb = GVP(edge_dims, edge_dims)
        self.convs = nn.ModuleList([GVPConv(hidden, hidden, edge_dims) for _ in range(layers)])
        self.out = nn.Linear(hidden[0], hidden[0])

    def forward(self, batch):
        h_s, h_v = self.node_emb((batch.node_s, batch.node_v))
        e_s, e_v = self.edge_emb((batch.edge_s, batch.edge_v))
        for conv in self.convs:
            ds, dv = conv((h_s, h_v), batch.edge_index, (e_s, e_v))
            h_s = h_s + ds
            h_v = h_v + dv if h_v is not None and dv is not None else h_v
        from torch_scatter import scatter_mean
        return self.out(scatter_mean(h_s, batch.batch, dim=0)), h_s


# =============================================================================
#Sequence encoder module
# =============================================================================

class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.Tanh(),
            nn.Linear(dim // 4, 1)
        )

    def forward(self, x, mask=None):
        scores = self.attn(x).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask, -1e4)
        weights = F.softmax(scores, dim=-1)
        return (weights.unsqueeze(-1) * x).sum(1)


class SeqEncoderV6(nn.Module):
    def __init__(self, in_dim=2560, hid=256, layers=2):
        super().__init__()
        self.proj = nn.Linear(in_dim, hid)
        self.tf = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hid, 8, hid * 4, 0.1, batch_first=True), layers)
        self.pool = AttentionPooling(hid)

    def forward(self, x, lens):
        x = self.proj(x)
        B, L, _ = x.shape
        mask = torch.arange(L, device=x.device).expand(B, L) >= lens.unsqueeze(1)
        x = self.tf(x, src_key_padding_mask=mask)
        global_repr = self.pool(x, mask)
        return global_repr, x


# =============================================================================
# Transformer classier
# =============================================================================

class TransformerClassifierV6(nn.Module):
    def __init__(self, seq_dim, struct_dim, d_model=128,
                 n_heads=4, n_layers=2, n_cls=6, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        self.seq_tokenizer = nn.Sequential(
            nn.LayerNorm(seq_dim),
            nn.Linear(seq_dim, d_model)
        )
        self.struct_tokenizer = nn.Sequential(
            nn.LayerNorm(struct_dim),
            nn.Linear(struct_dim, d_model)
        )
        self.cross_proj_seq = nn.Linear(seq_dim, d_model)
        self.cross_proj_struct = nn.Linear(struct_dim, d_model)

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, 4, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_model * 4, dropout, batch_first=True, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_cls)
        )

    def forward(self, seq_feat, struct_feat):
        B = seq_feat.size(0)

        tok_seq = self.seq_tokenizer(seq_feat).unsqueeze(1)
        tok_struct = self.struct_tokenizer(struct_feat).unsqueeze(1)
        cross = torch.tanh(self.cross_proj_seq(seq_feat)) * \
                torch.tanh(self.cross_proj_struct(struct_feat))
        tok_cross = cross.unsqueeze(1)
        cls = self.cls_token.expand(B, -1, -1)

        tokens = torch.cat([cls, tok_seq, tok_struct, tok_cross], dim=1)
        tokens = tokens + self.pos_embed

        tokens = self.transformer(tokens)
        cls_out = tokens[:, 0]
        return self.head(cls_out)


# =============================================================================
# Model
# =============================================================================

class CLEF_GVP_V6(nn.Module):
    def __init__(self, n_cls=6, esm_dim=2560, seq_hid=256, struct_hid=100,
                 cls_d_model=128, cls_n_heads=4, cls_n_layers=2):
        super().__init__()

        self.seq_enc = SeqEncoderV6(esm_dim, seq_hid)
        self.struct_enc = GVPEncoder(hidden=(struct_hid, 16))

        self.seq_norm = nn.LayerNorm(seq_hid)
        self.struct_norm = nn.LayerNorm(struct_hid)

        self.classifier = TransformerClassifierV6(
            seq_hid, struct_hid, cls_d_model, cls_n_heads, cls_n_layers, n_cls)

        proj_dim = 128
        self.seq_projector = nn.Sequential(
            nn.Linear(seq_hid, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim))
        self.struct_projector = nn.Sequential(
            nn.Linear(struct_hid, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim))

    def forward(self, batch):
        seq_global, _ = self.seq_enc(batch['esm_feature'], batch['valid_lens'])
        struct_global, _ = self.struct_enc(batch['graph_data'])

        seq_global = self.seq_norm(seq_global)
        struct_global = self.struct_norm(struct_global)

        logits = self.classifier(seq_global, struct_global)

        seq_proj = F.normalize(self.seq_projector(seq_global), dim=-1)
        struct_proj = F.normalize(self.struct_projector(struct_global), dim=-1)

        return logits, seq_proj, struct_proj


# =============================================================================
# =============================================================================

def pdb_to_graph(pdb_path: str) -> Optional[Data]:
    parser = PDBParser(QUIET=True)
    try:
        struct = parser.get_structure('p', pdb_path)
    except Exception as e:
        print(f"  Warning: Unable to resolvePDB file {pdb_path}: {e}")
        return None

    residues = [r for m in struct for c in m for r in c
                if is_aa(r, standard=True) and 'CA' in r]

    if len(residues) < 5:
        print(f"  Warning: The PDB file has too few residues ({len(residues)} < 5): {pdb_path}")
        return None

    node_s, node_v, ca_coords = [], [], []

    for res in residues:
        try:
            n = res['N'].get_coord()
            ca = res['CA'].get_coord()
            c = res['C'].get_coord()
            ca_coords.append(ca)

            v1, v2 = n - ca, c - ca
            angle = np.arccos(np.clip(
                np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8),
                -1, 1))

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

        except Exception:
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


def extract_esm_features(sequences: Dict[str, str], device: torch.device,
                         batch_size: int = 4) -> Dict[str, np.ndarray]:
    print("\nloadESM-2 (esm2_t36_3B_UR50D)...")
    
    import esm
    import gc
    
    model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
    model = model.to(device)
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    
    esm_features = {}
    seq_list = [(pid, seq) for pid, seq in sequences.items()]
    
    print(f"Extract ESM features from {len(seq_list)} sequences.")
    
    gc_interval = 50
    
    with torch.no_grad():
        for i in tqdm(range(0, len(seq_list), batch_size), desc="ESM feature extract"):
            batch_data = seq_list[i:i + batch_size]
            
            batch_data_truncated = [(pid, seq[:1022]) for pid, seq in batch_data]
            
            _, _, batch_tokens = batch_converter(batch_data_truncated)
            batch_tokens = batch_tokens.to(device)
            
            results = model(batch_tokens, repr_layers=[36], return_contacts=False)
            representations = results["representations"][36]
            
            for j, (pid, seq) in enumerate(batch_data_truncated):
                seq_repr = representations[j, 1:len(seq) + 1].cpu().half().numpy()
                esm_features[batch_data[j][0]] = seq_repr
            
            del batch_tokens, results, representations
            
            if (i // batch_size + 1) % gc_interval == 0:
                torch.cuda.empty_cache()
                gc.collect()
    
    del model, alphabet, batch_converter
    torch.cuda.empty_cache()
    gc.collect()
    
    print(f"  ✓ Feature extraction complete, totaling {len(esm_features)} sequences.")
    
    return esm_features


def load_sequences(fasta_path: str) -> Dict[str, str]:
    sequences = {}
    for record in SeqIO.parse(fasta_path, 'fasta'):
        sequences[record.id] = str(record.seq)
    return sequences


def load_pdb_files(pdb_dirs: List[str], protein_ids: List[str], 
                   cache_path: Optional[str] = None) -> Dict[str, Data]:


    if cache_path and os.path.exists(cache_path):
        print(f"\nLoad graph data from cache: {cache_path}")
        cached_data = torch.load(cache_path, map_location='cpu', weights_only=False)
        cached_graphs = cached_data.get('graphs', {})
        
        missing_in_cache = [pid for pid in protein_ids if pid not in cached_graphs]
        
        if not missing_in_cache:
            print(f"  ✓ {len(cached_graphs)} protein graphs were loaded from the cache.")
            return {pid: cached_graphs[pid] for pid in protein_ids if pid in cached_graphs}
        else:
            print(f"  The cache is missing {len(missing_in_cache)} proteins, so it will be reprocessed")
    
    if isinstance(pdb_dirs, str):
        pdb_dirs = [pdb_dirs]
    
    print(f"\nProcessing PDB files...")
    print(f"  Search Directory:")
    for d in pdb_dirs:
        print(f"    - {d}")
    

    pdb_files = {}
    for pdb_dir in pdb_dirs:
        pdb_dir = Path(pdb_dir)
        if not pdb_dir.exists():
            print(f" Warning: Directory does not exist, skip: {pdb_dir}")
            continue
        
        for f in pdb_dir.rglob("*.pdb"):
            pid = f.stem
            if pid not in pdb_files:
                pdb_files[pid] = str(f)
    
    
    graphs = {}
    missing = []
    
    for pid in tqdm(protein_ids, desc="Constructing a protein map"):
        candidates = [
            pid,
            pid.replace('|', '_'),
            pid.replace('~', '_'),
            pid.replace('/', '_'),
            pid.split('|')[0],
        ]
        
        found = False
        for candidate in candidates:
            if candidate in pdb_files:
                graph = pdb_to_graph(pdb_files[candidate])
                if graph is not None:
                    graphs[pid] = graph
                    found = True
                break
        
        if not found:
            missing.append(pid)
    
    if missing:
        print(f"  Warning: {len(missing)} proteins are missing PDB files.")
        if len(missing) <= 10:
            print(f"    missing: {missing}")
        else:
            print(f"    fast 10: {missing[:10]}...")
    
    if cache_path and graphs:
        cache_dir = os.path.dirname(cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        
        cache_data = {
            'graphs': graphs,
            'pdb_dir': str(pdb_dir),
            'num_proteins': len(graphs),
        }
        torch.save(cache_data, cache_path)
        print(f"  ✓ graphs_data was cached: {cache_path}")
    
    return graphs


def load_esm_features(sequences: Dict[str, str], device: torch.device,
                      cache_path: Optional[str] = None,
                      batch_size: int = 4) -> Dict[str, np.ndarray]:

    import gc
    

    if cache_path and os.path.exists(cache_path):
        print(f"\nLoad ESM features from cache: {cache_path}")
        try:
            cached_data = torch.load(cache_path, map_location='cpu', weights_only=False)
            cached_features = cached_data.get('features', {})
            

            missing_in_cache = [pid for pid in sequences.keys() if pid not in cached_features]
            
            if not missing_in_cache:
                print(f"  ✓ ESM features of {len(cached_features)} sequences were loaded from the cache.")
                result = {pid: cached_features[pid] for pid in sequences.keys() if pid in cached_features}
                del cached_data, cached_features
                gc.collect()
                return result
            else:
                print(f"  The cache is missing {len(missing_in_cache)} sequences; it will be retrieved again....")
                del cached_data, cached_features
                gc.collect()
        except Exception as e:
            print(f"  Warning: Loading cache failed ({e}), will fetch again...")
    

    esm_features = extract_esm_features(sequences, device, batch_size)
    
    if cache_path and esm_features:
        cache_dir = os.path.dirname(cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        
        print(f"  Save ESM feature cache...")
        try:
            cache_data = {
                'features': esm_features,
                'num_sequences': len(esm_features),
            }
            torch.save(cache_data, cache_path, pickle_protocol=4)
            print(f"  ✓ ESMfeature was cached: {cache_path}")
            del cache_data
            gc.collect()
        except Exception as e:
            print(f" Warning: Failed to save cache ({e})")
    
    return esm_features


def collate_fn(batch: List[dict]) -> Optional[dict]:
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    
    esm_list = [b['esm_feat'] for b in batch]
    max_len = max(e.size(0) for e in esm_list)
    feat_dim = esm_list[0].size(1)
    
    padded = torch.zeros(len(batch), max_len, feat_dim)
    lens = []
    
    for i, e in enumerate(esm_list):
        padded[i, :e.size(0)] = e
        lens.append(e.size(0))
    
    return {
        'esm_feature': padded,
        'valid_lens': torch.tensor(lens),
        'graph_data': Batch.from_data_list([b['graph'] for b in batch]),
        'id_list': [b['id'] for b in batch]
    }



class GenomePredictor:

    
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"use device: {self.device}")
        

        print(f"\nloading model: {model_path}")
        self.model = CLEF_GVP_V6(
            NUM_CLASSES, ESM_DIM, SEQ_HID, STRUCT_HID,
            CLS_D_MODEL, CLS_N_HEADS, CLS_N_LAYERS
        ).to(self.device)
        
        state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print("load model successful!")
    
    def predict(self, fasta_path: str, pdb_dirs: List[str],
                batch_size: int = 8, esm_batch_size: int = 2,
                cache_dir: Optional[str] = None) -> pd.DataFrame:


        if isinstance(pdb_dirs, str):
            pdb_dirs = [pdb_dirs]
        

        graph_cache_path = None
        esm_cache_path = None
        
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            fasta_name = Path(fasta_path).stem
            graph_cache_path = os.path.join(cache_dir, f"{fasta_name}_graphs.pt")
            esm_cache_path = os.path.join(cache_dir, f"{fasta_name}_esm.pt")
            print(f"\ncache dir: {cache_dir}")
        

        print(f"\n{'='*60}")
        print("Step 1/4: Load the FASTA sequence")
        print(f"{'='*60}")
        sequences = load_sequences(fasta_path)
        print(f"  load {len(sequences)}Seq")
        

        print(f"\n{'='*60}")
        print("Step 2/4: Processing PDB structure files")
        print(f"{'='*60}")
        graphs = load_pdb_files(pdb_dirs, list(sequences.keys()), cache_path=graph_cache_path)
        print(f"  Successfully obtained {len(graphs)} protein graphs")
        

        print(f"\n{'='*60}")
        print("Step 3/4: Extract ESM-2 sequence features")
        print(f"{'='*60}")

        valid_sequences = {pid: seq for pid, seq in sequences.items() if pid in graphs}
        esm_features = load_esm_features(valid_sequences, self.device, 
                                          cache_path=esm_cache_path,
                                          batch_size=esm_batch_size)
        print(f"  The features of {len(esm_features)} sequences were obtained.")
        

        del valid_sequences
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        

        print(f"\n{'='*60}")
        print("Step 4/4: Model Prediction")
        print(f"{'='*60}")
        

        valid_ids = sorted(set(graphs.keys()) & set(esm_features.keys()))
        print(f"  Valid protein count: {len(valid_ids)}")
        
        data_list = []
        for pid in valid_ids:
            esm_feat = torch.tensor(esm_features[pid], dtype=torch.float)
            graph = graphs[pid]
            
            min_len = min(esm_feat.size(0), graph.num_nodes)
            esm_feat = esm_feat[:min_len]
            
            data_list.append({
                'esm_feat': esm_feat,
                'graph': graph,
                'id': pid
            })
        
        del esm_features, graphs
        gc.collect()
        
        results = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(data_list), batch_size), desc="Forecast progress"):
                batch_data = data_list[i:i + batch_size]
                batch = collate_fn(batch_data)
                
                if batch is None:
                    continue
                

                batch['esm_feature'] = batch['esm_feature'].to(self.device)
                batch['valid_lens'] = batch['valid_lens'].to(self.device)
                batch['graph_data'] = batch['graph_data'].to(self.device)
                

                logits, _, _ = self.model(batch)
                probs = F.softmax(logits, dim=-1)
                preds = logits.argmax(dim=-1)
                

                for j, pid in enumerate(batch['id_list']):
                    pred_class = preds[j].item()
                    prob_values = probs[j].cpu().numpy()
                    
                    result = {
                        'protein_id': pid,
                        'sequence': sequences.get(pid, ''),
                        'sequence_length': len(sequences.get(pid, '')),
                        'predicted_class': CLASS_NAMES[pred_class],
                        'predicted_label': pred_class,
                        'confidence': prob_values[pred_class],
                    }
                    
                    for k, cn in enumerate(CLASS_NAMES):
                        result[f'prob_{cn}'] = prob_values[k]
                    
                    results.append(result)
        
        df = pd.DataFrame(results)
        
        df['is_effector'] = df['predicted_label'] > 0
        df = df.sort_values(['is_effector', 'confidence'], ascending=[False, False])
        df = df.drop('is_effector', axis=1)
        df = df.reset_index(drop=True)
        
        return df
    
    def summarize_results(self, df: pd.DataFrame) -> None:

        print(f"\n{'='*60}")
        print("Summary of Prediction Results")
        print(f"{'='*60}")
        
        print(f"\nTotal protein count: {len(df)}")
        print(f"\nNumber of predicted items for each category:")
        for cn in CLASS_NAMES:
            count = (df['predicted_class'] == cn).sum()
            pct = 100 * count / len(df) if len(df) > 0 else 0
            print(f"  {cn:<15}: {count:>5} ({pct:>5.1f}%)")
        
        
        effectors = df[df['predicted_label'] > 0]
        if len(effectors) > 0:
            print(f"\nPredicted effector proteins (total {len(effectors)}):")
            high_conf = effectors[effectors['confidence'] >= 0.8]
            print(f"  High confidence (≥0.8): {len(high_conf)}")
            
            if len(high_conf) > 0:
                print(f"\n  Top 10 High-confidence effector proteins:")
                for i, row in high_conf.head(10).iterrows():
                    print(f"    {row['protein_id']}: {row['predicted_class']} "
                          f"(Confidence: {row['confidence']:.3f})")




def main():
    parser = argparse.ArgumentParser(
        description='Genome-wide Effector Protein Prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用多个PDB目录（高/低置信度）
  python predict_genome.py \\
      --fasta ./Legionella_pneumophila.fasta \\
      --pdb_dirs ./pdb_output/pdb_high_confidence ./pdb_output/pdb_low_confidence \\
      --model_path ./results_v6/fold_5/best_model.pt \\
      --output ./predictions.csv \\
      --cache_dir ./cache/

  # 使用单个PDB目录
  python predict_genome.py \\
      --fasta ./Legionella_pneumophila.fasta \\
      --pdb_dirs ./pdb_structures/ \\
      --model_path ./results_v6/fold_5/best_model.pt \\
      --output ./predictions.csv

  # 后续运行（直接加载缓存，速度更快）
  python predict_genome.py \\
      --fasta ./Legionella_pneumophila.fasta \\
      --pdb_dirs ./pdb_output/pdb_high_confidence ./pdb_output/pdb_low_confidence \\
      --model_path ./results_v6/fold_5/best_model.pt \\
      --output ./predictions.csv \\
      --cache_dir ./cache/
        """
    )
    
    parser.add_argument('--fasta', required=True,
                        help='输入FASTA文件路径')
    parser.add_argument('--pdb_dirs', required=True, nargs='+',
                        help='PDB文件目录列表，可指定多个目录 (如 high_confidence low_confidence)。'
                             '优先使用前面目录中的文件。')
    parser.add_argument('--model_path', required=True,
                        help='训练好的模型权重路径 (.pt文件)')
    parser.add_argument('--output', default='predictions.csv',
                        help='输出CSV文件路径 (默认: predictions.csv)')
    parser.add_argument('--cache_dir', default=None,
                        help='缓存目录路径，用于存储预处理的图和ESM特征 (.pt文件)。'
                             '首次运行会生成缓存，后续运行直接加载以加速预测。')
    parser.add_argument('--device', default='cuda',
                        help='计算设备 (cuda/cpu, 默认: cuda)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='预测批次大小 (默认: 8)')
    parser.add_argument('--esm_batch_size', type=int, default=2,
                        help='ESM特征提取批次大小 (默认: 2, 根据GPU显存调整)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.fasta):
        raise FileNotFoundError(f"FASTA file does not exist: {args.fasta}")
    
    valid_pdb_dirs = [d for d in args.pdb_dirs if os.path.exists(d)]
    if not valid_pdb_dirs:
        raise FileNotFoundError(f"All PDB directories do not exist.: {args.pdb_dirs}")
    
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file does not exist: {args.model_path}")
    
    print("="*60)
    print("Genome-wide effector protein prediction")
    print("="*60)
    print(f"\ninput:")
    print(f"  FASTA: {args.fasta}")
    print(f"  PDBdir:")
    for i, d in enumerate(args.pdb_dirs):
        status = "✓" if os.path.exists(d) else "✗ (No ex)"
        priority = "(first)" if i == 0 else ""
        print(f"    {i+1}. {d} {status} {priority}")
    print(f"  model: {args.model_path}")
    print(f"  output: {args.output}")
    if args.cache_dir:
        print(f"  Cache directory: {args.cache_dir}")
    

    predictor = GenomePredictor(args.model_path, args.device)
    

    results_df = predictor.predict(
        args.fasta,
        args.pdb_dirs,
        batch_size=args.batch_size,
        esm_batch_size=args.esm_batch_size,
        cache_dir=args.cache_dir
    )
    
    results_df.to_csv(args.output, index=False)
    print(f"\nThe results have been saved to: {args.output}")
    
    predictor.summarize_results(results_df)
    
    print(f"\n{'='*60}")
    print("Prediction complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()