import os
import sys
import argparse
import json
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data, Batch
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from Bio import SeqIO
try:
    from Bio.PDB.Polypeptide import three_to_one
except ImportError:
    from Bio.SeqUtils import seq1 as three_to_one
import warnings


CLASS_NAMES = ['Non-Effector', 'T1SE', 'T2SE', 'T3SE', 'T4SE', 'T6SE']
AA_IDX = {a: i for i, a in enumerate('ACDEFGHIKLMNPQRSTVWYX')}

ESM_DIM = 2560
SEQ_HID = 256
STRUCT_HID = 100
CLS_D_MODEL = 128
CLS_N_HEADS = 4
CLS_N_LAYERS = 2
NUM_CLASSES = 6


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
            v = self.Wh(v.transpose(-1,-2)).transpose(-1,-2)
            s = torch.cat([s, torch.norm(v, dim=-1)], -1)
        s = self.Ws(s)
        if self.act_s: s = self.act_s(s)
        if self.vo > 0 and v is not None and self.act_v:
            v = v * self.act_v(torch.norm(v, dim=-1, keepdim=True))
        else: v = None
        return s, v


class GVPConv(nn.Module):
    def __init__(self, in_dims, out_dims, edge_dims):
        super().__init__()
        self.msg = GVP((2*in_dims[0]+edge_dims[0], 2*in_dims[1]+edge_dims[1]), out_dims)
    
    def forward(self, x, edge_index, edge_attr):
        s, v = x; es, ev = edge_attr; src, dst = edge_index
        ms = torch.cat([s[src], s[dst], es], -1)
        mv = torch.cat([v[src], v[dst], ev], 1) if v is not None else ev
        ms, mv = self.msg((ms, mv))
        from torch_scatter import scatter_mean
        return scatter_mean(ms, dst, dim=0, dim_size=s.size(0)), \
               scatter_mean(mv, dst, dim=0, dim_size=s.size(0)) if mv is not None else None


class GVPEncoder(nn.Module):
    def __init__(self, node_dims=(6,3), edge_dims=(32,1), hidden=(100,16), layers=3):
        super().__init__()
        self.node_emb = GVP(node_dims, hidden)
        self.edge_emb = GVP(edge_dims, edge_dims)
        self.convs = nn.ModuleList([GVPConv(hidden, hidden, edge_dims) for _ in range(layers)])
        self.out = nn.Linear(hidden[0], hidden[0])
    
    def forward(self, batch, return_node_features=False):
        h_s, h_v = self.node_emb((batch.node_s, batch.node_v))
        e_s, e_v = self.edge_emb((batch.edge_s, batch.edge_v))
        for conv in self.convs:
            ds, dv = conv((h_s, h_v), batch.edge_index, (e_s, e_v))
            h_s, h_v = h_s + ds, (h_v + dv if h_v is not None and dv is not None else h_v)
        from torch_scatter import scatter_mean
        node_features = self.out(h_s)
        global_features = scatter_mean(node_features, batch.batch, dim=0)
        if return_node_features:
            return global_features, node_features
        return global_features, h_s


class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(dim, dim // 4), nn.Tanh(), nn.Linear(dim // 4, 1))

    def forward(self, x, mask=None):
        scores = self.attn(x).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask, -1e4)
        weights = F.softmax(scores, dim=-1)
        return (weights.unsqueeze(-1) * x).sum(1), weights


class SeqEncoderV6(nn.Module):
    def __init__(self, in_dim=2560, hid=256, layers=2):
        super().__init__()
        self.proj = nn.Linear(in_dim, hid)
        self.tf = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hid, 8, hid*4, 0.1, batch_first=True), layers)
        self.pool = AttentionPooling(hid)

    def forward(self, x, lens):
        x = self.proj(x)
        B, L, _ = x.shape
        mask = torch.arange(L, device=x.device).expand(B, L) >= lens.unsqueeze(1)
        x = self.tf(x, src_key_padding_mask=mask)
        global_repr, attn_weights = self.pool(x, mask)
        return global_repr, x, attn_weights


class TransformerClassifierV6(nn.Module):
    def __init__(self, seq_dim, struct_dim, d_model=128, n_heads=4, n_layers=2, n_cls=6, dropout=0.1):
        super().__init__()
        self.seq_tokenizer = nn.Sequential(nn.LayerNorm(seq_dim), nn.Linear(seq_dim, d_model))
        self.struct_tokenizer = nn.Sequential(nn.LayerNorm(struct_dim), nn.Linear(struct_dim, d_model))
        self.cross_proj_seq = nn.Linear(seq_dim, d_model)
        self.cross_proj_struct = nn.Linear(struct_dim, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, 4, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4, dropout, batch_first=True, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, n_cls))

    def forward(self, seq_feat, struct_feat):
        B = seq_feat.size(0)
        tok_seq = self.seq_tokenizer(seq_feat).unsqueeze(1)
        tok_struct = self.struct_tokenizer(struct_feat).unsqueeze(1)
        cross = torch.tanh(self.cross_proj_seq(seq_feat)) * torch.tanh(self.cross_proj_struct(struct_feat))
        tok_cross = cross.unsqueeze(1)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tok_seq, tok_struct, tok_cross], dim=1) + self.pos_embed
        tokens = self.transformer(tokens)
        return self.head(tokens[:, 0])


class CLEF_GVP_V6(nn.Module):
    def __init__(self, n_cls=6, esm_dim=2560, seq_hid=256, struct_hid=100,
                 cls_d_model=128, cls_n_heads=4, cls_n_layers=2):
        super().__init__()
        self.seq_enc = SeqEncoderV6(esm_dim, seq_hid)
        self.struct_enc = GVPEncoder(hidden=(struct_hid, 16))
        self.seq_norm = nn.LayerNorm(seq_hid)
        self.struct_norm = nn.LayerNorm(struct_hid)
        self.classifier = TransformerClassifierV6(seq_hid, struct_hid, cls_d_model, cls_n_heads, cls_n_layers, n_cls)
        proj_dim = 128
        self.seq_projector = nn.Sequential(nn.Linear(seq_hid, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim))
        self.struct_projector = nn.Sequential(nn.Linear(struct_hid, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim))

    def forward(self, batch, return_attention=False, return_struct_weights=False):
        seq_global, seq_hidden, seq_attn = self.seq_enc(batch['esm_feature'], batch['valid_lens'])
        if return_struct_weights:
            struct_global, struct_node_features = self.struct_enc(batch['graph_data'], return_node_features=True)
        else:
            struct_global, _ = self.struct_enc(batch['graph_data'])
            struct_node_features = None
        seq_global = self.seq_norm(seq_global)
        struct_global = self.struct_norm(struct_global)
        logits = self.classifier(seq_global, struct_global)
        
        outputs = [logits]
        if return_attention:
            outputs.append(seq_attn)
        else:
            outputs.append(None)
        if return_struct_weights:
            outputs.append(struct_node_features)
        else:
            outputs.append(None)
        return tuple(outputs)

# =============================================================================
# Grad-CAM: Structure-branch Interpretability + PDB B-factor Visualization
# =============================================================================

class StructureGradCAM:
    """
    Grad-CAM for the GVP structure encoder branch.
    
    Algorithm (following Selvaraju et al., 2017, adapted from 2D feature maps
    to 1D per-node features on a protein graph):
    
      1. Forward hook on the LAST GVPConv layer captures node-level activations A_k
      2. Backward pass from the target-class logit yields gradients dY/dA_k
      3. Channel weights alpha_k = GlobalAvgPool(dY/dA_k)   [mean over nodes]
      4. Grad-CAM score per residue = ReLU( sum_k alpha_k * A_k )
      5. Normalize to [0, 1]
    
    High score => the GVP encoder relies heavily on that residue's 3D structural
    context to make its prediction.  This is NOT an attention heuristic; it is a
    true gradient-based attribution that is class-discriminative.
    """
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.activations = None
        self.gradients = None
        self._hooks = []
    
    def _register_hooks(self):
        """Register forward/backward hooks on the last GVPConv layer."""
        target_layer = self.model.struct_enc.convs[-1]
        
        def forward_hook(module, input, output):
            # output = (scalar_features, vector_features)
            self.activations = output[0].detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        h1 = target_layer.register_forward_hook(forward_hook)
        h2 = target_layer.register_full_backward_hook(backward_hook)
        self._hooks = [h1, h2]
    
    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []
    
    def compute(self, batch, target_class=None):
        """
        Compute Grad-CAM heatmap over residues.
        
        Args:
            batch: model input dict
            target_class: int or None (None = use predicted class)
        
        Returns:
            gradcam_scores: np.ndarray (num_residues,), normalized to [0,1]
            predicted_class: int
            probabilities: np.ndarray (num_classes,)
        """
        self._register_hooks()
        
        # --- Forward (with gradients enabled) ---
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(True)
        
        esm_feat = batch['esm_feature'].clone().detach().requires_grad_(False)
        valid_lens = batch['valid_lens']
        graph_data = batch['graph_data']
        
        # Make sure graph node features can propagate gradients
        graph_data.node_s = graph_data.node_s.clone().detach().requires_grad_(True)
        
        seq_global, seq_hidden, seq_attn = self.model.seq_enc(esm_feat, valid_lens)
        struct_global, struct_node_feat = self.model.struct_enc(graph_data, return_node_features=True)
        
        seq_global = self.model.seq_norm(seq_global)
        struct_global = self.model.struct_norm(struct_global)
        logits = self.model.classifier(seq_global, struct_global)
        probs = F.softmax(logits, dim=-1)
        
        pred_idx = logits.argmax(dim=-1).item()
        if target_class is None:
            target_class = pred_idx
        
        # --- Backward ---
        self.model.zero_grad()
        target_score = logits[0, target_class]
        target_score.backward(retain_graph=False)
        
        # --- Compute Grad-CAM ---
        if self.gradients is not None and self.activations is not None:
            # alpha_k = mean gradient over all nodes (global-average-pooling of gradients)
            alpha = self.gradients.mean(dim=0, keepdim=True)   # (1, hidden_dim)
            # Weighted combination: sum_k( alpha_k * A_k )
            cam = (alpha * self.activations).sum(dim=-1)       # (num_nodes,)
            cam = F.relu(cam)
            cam = cam.cpu().numpy()
            # Normalize
            if cam.max() > 0:
                cam = cam / cam.max()
        else:
            warnings.warn("[WARN] Grad-CAM hooks did not capture activations/gradients, returning zeros")
            num_nodes = graph_data.num_nodes if hasattr(graph_data, 'num_nodes') else 0
            cam = np.zeros(num_nodes)
        
        # Restore eval state
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()
        self._remove_hooks()
        
        return cam, pred_idx, probs.detach().cpu().numpy()[0]


def write_gradcam_pdb(input_pdb_path, output_pdb_path, gradcam_scores, 
                       top_k=15, neighbor_radius=8.0, sequence=None):
    """
    Write Grad-CAM scores into the B-factor column of a PDB file.
    
    Strategy:
      1. Top-K residues get their raw Grad-CAM score (normalized to 0-99.99)
      2. Spatial neighbors within `neighbor_radius` Angstroms get distance-
         decayed scores (closer = higher)
      3. All other residues get B-factor = 0 (white in PyMOL)
    
    In PyMOL:  spectrum b, white_red
      - Key pockets / domains => bright red
      - Surrounding region    => gradient (light pink)
      - Irrelevant residues   => white
    """
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure('gradcam', input_pdb_path)
    
    # Collect standard residues + CA coords
    residues = []
    ca_coords = []
    for model in struct:
        for chain in model:
            for res in chain:
                if is_aa(res, standard=True) and 'CA' in res:
                    residues.append(res)
                    ca_coords.append(res['CA'].get_coord())
    
    ca_coords = np.array(ca_coords)
    num_res = len(residues)
    
    # Align Grad-CAM scores to PDB residues
    n_scores = min(len(gradcam_scores), num_res)
    scores = np.zeros(num_res)
    scores[:n_scores] = gradcam_scores[:n_scores]
    
    # --- Find Top-K ---
    top_k_actual = min(top_k, n_scores)
    top_indices = np.argsort(scores)[-top_k_actual:][::-1]
    
    # --- Assign B-factors ---
    bfactors = np.zeros(num_res)
    
    # Top-K: direct score
    for idx in top_indices:
        bfactors[idx] = scores[idx]
    
    # Spatial neighbors: distance-decayed score
    for idx in top_indices:
        dists = np.linalg.norm(ca_coords - ca_coords[idx], axis=1)
        neighbors = np.where((dists > 0) & (dists <= neighbor_radius))[0]
        for nb in neighbors:
            decay = 1.0 - (dists[nb] / neighbor_radius)
            neighbor_score = scores[idx] * decay * 0.7
            bfactors[nb] = max(bfactors[nb], neighbor_score)
    
    # Normalize to 0 ~ 99.99 (PDB B-factor range)
    if bfactors.max() > 0:
        bfactors = (bfactors / bfactors.max()) * 99.99
    
    # --- Write into PDB ---
    # Reset all atoms to 0
    for model in struct:
        for chain in model:
            for res in chain:
                for atom in res:
                    atom.set_bfactor(0.0)
    
    # Set computed B-factors
    for i, res in enumerate(residues):
        for atom in res:
            atom.set_bfactor(float(bfactors[i]))
    
    # Save
    from Bio.PDB import PDBIO
    io = PDBIO()
    io.set_structure(struct)
    io.save(output_pdb_path)
    
    # --- Build report ---
    report = {
        'top_residues': [],
        'total_highlighted': int(np.sum(bfactors > 0)),
        'total_residues': num_res,
        'coverage_pct': float(np.sum(bfactors > 0) / num_res * 100),
    }
    
    for rank, idx in enumerate(top_indices):
        res = residues[idx]
        res_name = res.get_resname()
        res_id = res.get_id()[1]
        chain_id = res.get_parent().get_id()
        try:
            one_letter = three_to_one(res_name)
        except:
            one_letter = 'X'
        
        dists = np.linalg.norm(ca_coords - ca_coords[idx], axis=1)
        n_neighbors = int(np.sum((dists > 0) & (dists <= neighbor_radius)))
        
        report['top_residues'].append({
            'rank': rank + 1,
            'chain': chain_id,
            'residue_id': res_id,
            'residue_name': res_name,
            'one_letter': one_letter,
            'gradcam_score': float(scores[idx]),
            'bfactor': float(bfactors[idx]),
            'n_neighbors': n_neighbors,
        })
    
    return report


def write_pymol_script(pdb_path, script_path, report, protein_name="protein"):
    """
    Generate a PyMOL visualization script.
    
    IMPORTANT: The .pml content is PURE ASCII to avoid Windows GBK encoding
    errors.  All comments are in English.
    
    Usage in PyMOL:   @script_path
    Or manually:
        1. load your_gradcam.pdb
        2. spectrum b, white_red
    """
    top_residue_ids = [r['residue_id'] for r in report['top_residues']]
    top_resi_str = '+'.join(str(r) for r in top_residue_ids)
    
    script = f"""# ================================================================
# CLEF-GVP Grad-CAM Structure Visualization Script (PyMOL)
# ================================================================
# Usage: In PyMOL command line, type   @{os.path.basename(script_path)}
#        Or: File -> Run Script -> select this .pml file
# ================================================================

# Load the Grad-CAM annotated PDB
load {os.path.basename(pdb_path)}, {protein_name}

# Basic display settings
hide everything
show cartoon, {protein_name}
set cartoon_transparency, 0.1
bg_color white

# ========== Core visualization: color by B-factor ==========
# The B-factor column has been replaced with Grad-CAM scores:
#   0      = no contribution   (white)
#   99.99  = highest importance (red)
spectrum b, white_red, {protein_name}

# ========== Highlight Top-{len(report['top_residues'])} key residues ==========
select hotspot, resi {top_resi_str}
show sticks, hotspot
set stick_radius, 0.2, hotspot

# Labels: show residue name + number on CA atoms
label hotspot and name CA, "%s%s" % (resn, resi)
set label_size, 14
set label_color, black
set label_font_id, 7

# ========== Surface display (optional, shows pocket shape) ==========
# Uncomment the following lines to show a semi-transparent surface:
# show surface, {protein_name}
# set transparency, 0.7

# ========== Camera setup ==========
orient
zoom hotspot, 15
set ray_shadow, 0
set antialias, 2
set ray_trace_mode, 1

# ========== Export high-quality image (optional) ==========
# ray 2400, 2400
# png gradcam_structure.png, dpi=300

# ================================================================
# Top key residues identified by Grad-CAM:
"""
    for r in report['top_residues']:
        script += f"# Rank {r['rank']}: Chain {r['chain']} {r['residue_name']}{r['residue_id']} "
        script += f"({r['one_letter']}) GradCAM={r['gradcam_score']:.4f} "
        script += f"Neighbors={r['n_neighbors']}\n"
    
    script += f"""#
# Coverage: {report['total_highlighted']}/{report['total_residues']} residues highlighted ({report['coverage_pct']:.1f}%)
# ================================================================
"""
    
    # Write with ASCII encoding to guarantee Windows compatibility
    with open(script_path, 'w', encoding='ascii', errors='replace') as f:
        f.write(script)


# =============================================================================
# 辅助函数
# =============================================================================

def pdb_to_graph(pdb_path):
    """将PDB文件转换为图数据"""
    parser = PDBParser(QUIET=True)
    try:
        struct = parser.get_structure('p', pdb_path)
    except Exception as e:
        raise ValueError(f"无法解析PDB文件: {e}")
    
    residues = [r for m in struct for c in m for r in c if is_aa(r, standard=True) and 'CA' in r]
    if len(residues) < 5:
        raise ValueError(f"PDB残基数量不足 ({len(residues)} < 5)")
    
    node_s, node_v, ca_coords = [], [], []
    for res in residues:
        try:
            n, ca, c = res['N'].get_coord(), res['CA'].get_coord(), res['C'].get_coord()
            ca_coords.append(ca)
            v1, v2 = n-ca, c-ca
            angle = np.arccos(np.clip(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-8),-1,1))
            aa_idx = AA_IDX.get(three_to_one(res.get_resname()), 20)
            node_s.append([np.linalg.norm(c-n)/10, np.linalg.norm(ca-n)/10, np.linalg.norm(c-ca)/10,
                          angle/np.pi, np.sin(angle), aa_idx/20])
            u1 = (c-ca)/(np.linalg.norm(c-ca)+1e-8)
            u2 = np.cross(n-ca, c-ca); u2 /= (np.linalg.norm(u2)+1e-8)
            node_v.append([u1, u2, np.cross(u1, u2)])
        except:
            node_s.append([0]*6)
            node_v.append([[0,0,0]]*3)
            ca_coords.append([0,0,0])
    
    ca_coords = np.array(ca_coords)
    N = len(ca_coords)
    diff = ca_coords[:,None]-ca_coords[None,:]
    dist = np.sqrt(np.sum(diff**2, axis=-1))
    mask = (dist < 10) & (dist > 0)
    for i in range(N-1):
        mask[i,i+1] = mask[i+1,i] = True
    src, dst = np.where(mask)
    if len(src) == 0:
        raise ValueError("无法构建图的边")
    
    d = dist[src, dst]
    edge_s = np.exp(-((d[:,None]-np.linspace(0,20,32))**2)/2).astype(np.float32)
    direction = diff[src, dst]
    direction /= (np.linalg.norm(direction, axis=-1, keepdims=True)+1e-8)
    
    return Data(
        node_s=torch.tensor(node_s, dtype=torch.float),
        node_v=torch.tensor(node_v, dtype=torch.float),
        edge_index=torch.tensor([src, dst], dtype=torch.long),
        edge_s=torch.tensor(edge_s),
        edge_v=torch.tensor(direction[:,None,:].astype(np.float32)),
        num_nodes=N
    )


def read_fasta(fasta_path):
    """读取FASTA文件，返回第一条序列"""
    for record in SeqIO.parse(fasta_path, 'fasta'):
        return str(record.seq)
    raise ValueError(f"FASTA文件为空: {fasta_path}")


# =============================================================================
# 预测器类
# =============================================================================

class CLEFGVPPredictor:
    def __init__(self, model_path, device=None, esm_model_name="esm2_t36_3B_UR50D"):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[INFO] 使用设备: {self.device}")
        
        # 加载CLEF-GVP模型
        print(f"[INFO] 加载CLEF-GVP模型: {model_path}")
        self.model = CLEF_GVP_V6(NUM_CLASSES, ESM_DIM, SEQ_HID, STRUCT_HID,
                                  CLS_D_MODEL, CLS_N_HEADS, CLS_N_LAYERS)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()
        
        # 加载ESM-2模型
        print(f"[INFO] 加载ESM-2模型: {esm_model_name} (首次可能需要下载)")
        import esm
        self.esm_model, self.esm_alphabet = esm.pretrained.load_model_and_alphabet(esm_model_name)
        self.esm_model.to(self.device)
        self.esm_model.eval()
        self.esm_batch_converter = self.esm_alphabet.get_batch_converter()
        print("[INFO] 模型加载完成!")
    
    def extract_esm_features(self, sequence):
        data = [("protein", sequence)]
        _, _, batch_tokens = self.esm_batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)
        with torch.no_grad():
            results = self.esm_model(batch_tokens, repr_layers=[36], return_contacts=False)
        return results["representations"][36][0, 1:-1, :].cpu().numpy()
    
    def predict(self, sequence, pdb_path, return_attention=True, return_struct_weights=True):
        # 提取ESM特征
        esm_feat = self.extract_esm_features(sequence)
        esm_tensor = torch.tensor(esm_feat, dtype=torch.float).unsqueeze(0)
        
        # 构建图
        graph = pdb_to_graph(pdb_path)
        
        # 对齐长度
        seq_len, graph_len = esm_tensor.size(1), graph.num_nodes
        min_len = min(seq_len, graph_len)
        esm_tensor = esm_tensor[:, :min_len, :]
        
        # 准备batch
        batch = {
            'esm_feature': esm_tensor.to(self.device),
            'valid_lens': torch.tensor([min_len]).to(self.device),
            'graph_data': Batch.from_data_list([graph]).to(self.device)
        }
        
        # 预测
        with torch.no_grad():
            logits, seq_attn, struct_weights = self.model(batch, 
                                                          return_attention=return_attention,
                                                          return_struct_weights=return_struct_weights)
            probs = F.softmax(logits, dim=-1)
        
        pred_idx = probs.argmax(dim=-1).item()
        probs_np = probs.cpu().numpy()[0]
        
        result = {
            'predicted_class': CLASS_NAMES[pred_idx],
            'class_index': pred_idx,
            'confidence': float(probs_np[pred_idx]),
            'probabilities': {CLASS_NAMES[i]: float(probs_np[i]) for i in range(NUM_CLASSES)},
        }
        
        # 保存序列注意力分数
        if return_attention and seq_attn is not None:
            attn = seq_attn.cpu().numpy()[0][:min_len]
            result['sequence_attention'] = attn.tolist()
            top_k = min(10, len(attn))
            top_idx = np.argsort(attn)[-top_k:][::-1]
            result['top_attention_residues'] = [
                {'position': int(i)+1, 'residue': sequence[i] if i < len(sequence) else '?', 
                 'attention': float(attn[i])}
                for i in top_idx
            ]
        
        # 保存结构权重
        if return_struct_weights and struct_weights is not None:
            struct_w = struct_weights.cpu().numpy()[:min_len]
            result['structure_weights'] = struct_w.tolist()
            # 计算每个残基的结构重要性(使用L2范数)
            struct_importance = np.linalg.norm(struct_w, axis=1)
            result['structure_importance'] = struct_importance.tolist()
            top_k_struct = min(10, len(struct_importance))
            top_struct_idx = np.argsort(struct_importance)[-top_k_struct:][::-1]
            result['top_structure_residues'] = [
                {'position': int(i)+1, 'residue': sequence[i] if i < len(sequence) else '?',
                 'importance': float(struct_importance[i])}
                for i in top_struct_idx
            ]
        
        return result

    # -----------------------------------------------------------------
    # Grad-CAM analysis: structure-branch interpretability
    # -----------------------------------------------------------------
    def run_gradcam(self, sequence, pdb_path, target_class=None,
                    output_pdb=None, output_script=None,
                    top_k=15, neighbor_radius=8.0):
        """
        Run Grad-CAM on the GVP structure branch and produce:
          - A new PDB with B-factors replaced by Grad-CAM scores
          - A PyMOL .pml script for one-click visualization
          - Raw Grad-CAM scores as .npy
        """
        print("\n[GradCAM] Starting structure Grad-CAM analysis...")
        
        # Extract ESM features
        esm_feat = self.extract_esm_features(sequence)
        esm_tensor = torch.tensor(esm_feat, dtype=torch.float).unsqueeze(0)
        
        # Build graph
        graph = pdb_to_graph(pdb_path)
        
        # Align lengths
        seq_len, graph_len = esm_tensor.size(1), graph.num_nodes
        min_len = min(seq_len, graph_len)
        esm_tensor = esm_tensor[:, :min_len, :]
        
        batch = {
            'esm_feature': esm_tensor.to(self.device),
            'valid_lens': torch.tensor([min_len]).to(self.device),
            'graph_data': Batch.from_data_list([graph]).to(self.device)
        }
        
        # Run Grad-CAM
        gradcam = StructureGradCAM(self.model, self.device)
        cam_scores, pred_idx, probs = gradcam.compute(batch, target_class=target_class)
        
        print(f"[GradCAM] Predicted: {CLASS_NAMES[pred_idx]} (confidence: {probs[pred_idx]:.4f})")
        print(f"[GradCAM] Score range: [{cam_scores.min():.4f}, {cam_scores.max():.4f}]")
        print(f"[GradCAM] Non-zero residues: {np.sum(cam_scores > 0.01)}/{len(cam_scores)}")
        
        # Default output paths
        base_name = os.path.splitext(os.path.basename(pdb_path))[0]
        if output_pdb is None:
            output_pdb = f"./{base_name}_gradcam.pdb"
        if output_script is None:
            output_script = f"./{base_name}_pymol.pml"
        
        # Write PDB with B-factors
        print(f"[GradCAM] Writing Grad-CAM PDB: {output_pdb}")
        report = write_gradcam_pdb(pdb_path, output_pdb, cam_scores,
                                    top_k=top_k, neighbor_radius=neighbor_radius,
                                    sequence=sequence)
        
        # Generate PyMOL script (pure ASCII, Windows-safe)
        print(f"[GradCAM] Writing PyMOL script: {output_script}")
        write_pymol_script(output_pdb, output_script, report, protein_name=base_name)
        
        # Save raw scores
        scores_path = output_pdb.replace('.pdb', '_scores.npy')
        np.save(scores_path, cam_scores)
        print(f"[GradCAM] Raw scores saved: {scores_path}")
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"Grad-CAM Structure Interpretability Report")
        print(f"{'='*60}")
        actual_target = target_class if target_class is not None else pred_idx
        print(f"  Target class   : {CLASS_NAMES[actual_target]}")
        print(f"  Total residues : {report['total_residues']}")
        print(f"  Highlighted    : {report['total_highlighted']} ({report['coverage_pct']:.1f}%)")
        print(f"\n  Top-{top_k} key residues (Grad-CAM hotspots):")
        print(f"  {'Rank':<5} {'Chain':<6} {'Residue':<10} {'Score':<10} {'Neighbors':<10}")
        print(f"  {'-'*41}")
        for r in report['top_residues']:
            print(f"  {r['rank']:<5} {r['chain']:<6} {r['one_letter']}{r['residue_id']:<8} "
                  f"{r['gradcam_score']:<10.4f} {r['n_neighbors']:<10}")
        
        print(f"\n  PyMOL quick start:")
        print(f"  1. Open PyMOL")
        print(f"  2. Drag {output_pdb} into PyMOL window")
        print(f"  3. Type:  spectrum b, white_red")
        print(f"     Or run script: @{output_script}")
        print(f"  4. White = no contribution, Red = key structural feature")
        print(f"{'='*60}")
        
        return {
            'gradcam_scores': cam_scores.tolist(),
            'predicted_class': CLASS_NAMES[pred_idx],
            'confidence': float(probs[pred_idx]),
            'probabilities': {CLASS_NAMES[i]: float(probs[i]) for i in range(NUM_CLASSES)},
            'report': report,
            'output_pdb': output_pdb,
            'output_script': output_script,
            'output_scores': scores_path,
        }


# =============================================================================
# 命令行接口
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='CLEF-GVP v6 效应蛋白预测工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # 必需参数
    parser.add_argument('--model', '-m', required=True, help='模型权重文件路径 (.pt)')
    
    # 单样本预测
    parser.add_argument('--pdb', '-p', help='PDB结构文件路径')
    parser.add_argument('--seq', '-s', help='蛋白质序列字符串')
    parser.add_argument('--fasta', '-f', help='FASTA序列文件路径')
    
    # 批量预测
    parser.add_argument('--input', '-i', help='批量输入CSV文件 (列: id,sequence,pdb_path)')
    parser.add_argument('--output', '-o', help='输出文件路径 (CSV或JSON)')
    
    # 可选参数
    parser.add_argument('--gpu', '-g', type=int, default=None, help='GPU编号 (默认自动选择)')
    parser.add_argument('--no-attention', action='store_true', help='不输出注意力分数')
    parser.add_argument('--no-struct-weights', action='store_true', help='不输出结构权重')
    parser.add_argument('--save-attention', help='保存序列注意力分数到文件 (numpy .npy格式)')
    parser.add_argument('--save-struct-weights', help='保存结构权重到文件 (numpy .npy格式)')
    parser.add_argument('--format', choices=['json', 'csv', 'text'], default='text', help='输出格式 (默认text)')
    parser.add_argument('--esm-model', default='esm2_t36_3B_UR50D', help='ESM模型名称')
    
    # Grad-CAM visualization arguments
    parser.add_argument('--gradcam', action='store_true', 
                        help='Run structure Grad-CAM analysis, produce PyMOL PDB')
    parser.add_argument('--gradcam-pdb', default=None,
                        help='Grad-CAM output PDB path (default: <input>_gradcam.pdb)')
    parser.add_argument('--gradcam-script', default=None,
                        help='PyMOL script path (default: <input>_pymol.pml)')
    parser.add_argument('--gradcam-topk', type=int, default=15,
                        help='Number of top residues to highlight (default: 15)')
    parser.add_argument('--gradcam-radius', type=float, default=8.0,
                        help='Spatial neighbor search radius in Angstroms (default: 8.0)')
    parser.add_argument('--gradcam-target', type=int, default=None,
                        help='Grad-CAM target class index (default: use predicted class)')
    
    args = parser.parse_args()
    
    # 设置设备
    if args.gpu is not None:
        device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    else:
        device = None
    
    # 初始化预测器
    predictor = CLEFGVPPredictor(args.model, device=device, esm_model_name=args.esm_model)
    
    return_attention = not args.no_attention
    return_struct_weights = not args.no_struct_weights
    
    # ========== 批量预测模式 ==========
    if args.input:
        print(f"\n[INFO] 批量预测模式: {args.input}")
        results = []
        
        with open(args.input, 'r') as f:
            reader = csv.DictReader(f)
            samples = list(reader)
        
        print(f"[INFO] 共 {len(samples)} 个样本")
        
        for i, sample in enumerate(samples):
            sid = sample.get('id', f'sample_{i+1}')
            seq = sample['sequence']
            pdb = sample['pdb_path']
            
            print(f"[{i+1}/{len(samples)}] 预测 {sid}...", end=' ')
            try:
                result = predictor.predict(seq, pdb, 
                                          return_attention=return_attention,
                                          return_struct_weights=return_struct_weights)
                result['id'] = sid
                results.append(result)
                print(f"→ {result['predicted_class']} ({result['confidence']:.3f})")
            except Exception as e:
                print(f"→ 错误: {e}")
                results.append({'id': sid, 'error': str(e)})
        
        # 保存结果
        output_path = args.output or 'predictions.csv'
        if output_path.endswith('.json'):
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
        else:
            with open(output_path, 'w', newline='') as f:
                fieldnames = ['id', 'predicted_class', 'confidence'] + CLASS_NAMES + ['error']
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                for r in results:
                    row = {'id': r.get('id'), 'predicted_class': r.get('predicted_class'),
                           'confidence': r.get('confidence'), 'error': r.get('error', '')}
                    if 'probabilities' in r:
                        row.update(r['probabilities'])
                    writer.writerow(row)
        
        print(f"\n[INFO] 结果已保存至: {output_path}")
        return
    
    # ========== 单样本预测模式 ==========
    if not args.pdb:
        parser.error("单样本预测需要 --pdb 参数")
    
    # 获取序列
    if args.seq:
        sequence = args.seq
    elif args.fasta:
        sequence = read_fasta(args.fasta)
    else:
        parser.error("需要提供序列: --seq 或 --fasta")
    
    print(f"\n[INFO] 单样本预测")
    print(f"  PDB: {args.pdb}")
    print(f"  序列长度: {len(sequence)}")
    
    # 预测
    result = predictor.predict(sequence, args.pdb, 
                              return_attention=return_attention,
                              return_struct_weights=return_struct_weights)
    
    # 保存注意力分数到文件
    if args.save_attention and 'sequence_attention' in result:
        attn_array = np.array(result['sequence_attention'])
        np.save(args.save_attention, attn_array)
        print(f"[INFO] 序列注意力分数已保存至: {args.save_attention}")
    
    # 保存结构权重到文件
    if args.save_struct_weights and 'structure_weights' in result:
        struct_array = np.array(result['structure_weights'])
        np.save(args.save_struct_weights, struct_array)
        print(f"[INFO] 结构权重已保存至: {args.save_struct_weights}")
        # 同时保存结构重要性分数
        if 'structure_importance' in result:
            importance_path = args.save_struct_weights.replace('.npy', '_importance.npy')
            importance_array = np.array(result['structure_importance'])
            np.save(importance_path, importance_array)
            print(f"[INFO] 结构重要性分数已保存至: {importance_path}")
    
    # ========== Grad-CAM structure visualization ==========
    if args.gradcam:
        gradcam_result = predictor.run_gradcam(
            sequence, args.pdb,
            target_class=args.gradcam_target,
            output_pdb=args.gradcam_pdb,
            output_script=args.gradcam_script,
            top_k=args.gradcam_topk,
            neighbor_radius=args.gradcam_radius
        )
        result['gradcam'] = {
            'scores': gradcam_result['gradcam_scores'],
            'report': gradcam_result['report'],
            'output_pdb': gradcam_result['output_pdb'],
            'output_script': gradcam_result['output_script'],
        }
    
    # 输出结果
    if args.format == 'json':
        output = json.dumps(result, indent=2)
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output)
            print(f"[INFO] 结果已保存至: {args.output}")
        else:
            print(output)
    
    elif args.format == 'csv':
        output_path = args.output or 'prediction.csv'
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['predicted_class', 'confidence'] + CLASS_NAMES)
            row = [result['predicted_class'], result['confidence']]
            row += [result['probabilities'][c] for c in CLASS_NAMES]
            writer.writerow(row)
        print(f"[INFO] 结果已保存至: {output_path}")
    
    else:  # text
        print("\n" + "="*60)
        print("预测结果")
        print("="*60)
        print(f"  预测类别: {result['predicted_class']}")
        print(f"  置信度:   {result['confidence']:.4f}")
        print("\n各类别概率:")
        for cls, prob in result['probabilities'].items():
            bar = '█' * int(prob * 30)
            print(f"  {cls:<14} {prob:.4f} {bar}")
        
        if 'top_attention_residues' in result:
            print("\n高注意力残基 (Top 10):")
            print(f"  {'位置':<6} {'残基':<6} {'注意力':<10}")
            print("  " + "-"*24)
            for r in result['top_attention_residues']:
                print(f"  {r['position']:<6} {r['residue']:<6} {r['attention']:.6f}")
        
        if 'top_structure_residues' in result:
            print("\n高结构重要性残基 (Top 10):")
            print(f"  {'位置':<6} {'残基':<6} {'重要性':<10}")
            print("  " + "-"*24)
            for r in result['top_structure_residues']:
                print(f"  {r['position']:<6} {r['residue']:<6} {r['importance']:.6f}")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\n[INFO] 完整结果已保存至: {args.output}")


if __name__ == "__main__":
    main()
