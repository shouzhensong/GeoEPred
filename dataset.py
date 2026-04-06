import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch


class GVPDataset(Dataset):
    """Dataset that pairs ESM features, structural graphs, and labels.

    Each sample contains:
        - ``esm_feat``:  Residue-level ESM-2 embeddings ``(L, D)``.
        - ``graph``:     PyG ``Data`` object with geometric node/edge features.
        - ``label``:     Integer class label.
        - ``id``:        Protein identifier string.

    The dataset only yields samples whose protein ID exists in all three
    data sources (ESM, graphs, labels).

    Args:
        esm_feat:  ``dict[str, np.ndarray]`` — protein ID → ESM embedding.
        labels:    ``dict[str, int]`` — protein ID → class label.
        graphs:    ``dict[str, Data]`` — protein ID → PyG graph.
        ids:       ``list[str]`` — ordered list of valid protein IDs.
    """

    def __init__(self, esm_feat: dict, labels: dict, graphs: dict, ids: list):
        self.esm = esm_feat
        self.labels = labels
        self.graphs = graphs
        self.ids = ids

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int) -> dict:
        pid = self.ids[i]
        e = torch.tensor(self.esm[pid], dtype=torch.float)
        g = self.graphs[pid]
        # Align ESM length with graph node count
        if e.size(0) != g.num_nodes:
            e = e[: min(e.size(0), g.num_nodes)]
        return {
            "esm_feat": e,
            "graph": g,
            "label": torch.tensor(self.labels[pid]),
            "id": pid,
        }


def collate_fn(batch: list[dict]) -> dict | None:
    """Custom collation function.

    Pads ESM embeddings to the longest sequence in the batch and
    assembles PyG graphs into a single ``Batch`` object.

    Args:
        batch: List of sample dicts from ``GVPDataset.__getitem__``.

    Returns:
        Collated dictionary with keys:
            - ``esm_feature``:  ``(B, max_len, D)``
            - ``valid_lens``:   ``(B,)``
            - ``graph_data``:   ``torch_geometric.data.Batch``
            - ``label``:        ``(B,)``
            - ``id_list``:      ``list[str]``

        Returns ``None`` if all samples are invalid.
    """
    batch = [b for b in batch if b]
    if not batch:
        return None

    esm_list = [b["esm_feat"] for b in batch]
    max_len = max(e.size(0) for e in esm_list)
    feat_dim = esm_list[0].size(1)

    padded = torch.zeros(len(batch), max_len, feat_dim)
    lens = []
    for i, e in enumerate(esm_list):
        padded[i, : e.size(0)] = e
        lens.append(e.size(0))

    return {
        "esm_feature": padded,
        "valid_lens": torch.tensor(lens),
        "graph_data": Batch.from_data_list([b["graph"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "id_list": [b["id"] for b in batch],
    }
