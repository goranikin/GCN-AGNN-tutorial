"""
GCN vs AGNN (DIFUSCO-style) for TSP Edge Classification
========================================================
This script implements two GNN architectures and trains them on TSP data:

1. GCN (Kipf & Welling, ICLR 2017)
   - Propagation rule: H^(l+1) = σ(D̃^{-1/2} Ã D̃^{-1/2} H^(l) W^(l))
   - Originally designed for node classification
   - We adapt it for edge classification by concatenating node pairs

2. AGNN - Anisotropic Graph Neural Network (used in DIFUSCO, NeurIPS 2023)
   - Maintains BOTH node embeddings h_i AND edge embeddings e_ij
   - Uses learnable directional projections (P, Q, R, U, V matrices)
   - Gating mechanism with sigmoid + Hadamard product

Dataset: TSP instances with LKH-optimal tours
Task: Binary edge classification — predict which edges belong to the optimal tour

Usage:
    python gcn_vs_agnn_tsp.py --data_path <path_to_tsp_file> --num_nodes 50
"""

import argparse
import math
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Part 1: DATA LOADING
# ============================================================

class TSPDataset(Dataset):
    """
    Parses TSP data in the standard format:
        x0 y0 x1 y1 ... xN yN output n0 n1 n2 ... nN n0

    Each line = one TSP instance with coordinates and optimal tour.
    We convert the tour into binary edge labels on a complete graph.
    """

    def __init__(self, file_path: str, num_nodes: int = 50):
        self.num_nodes = num_nodes
        self.instances = []
        self.edge_labels = []

        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Split at "output" to separate coordinates from tour
                parts = line.split("output")
                coord_str = parts[0].strip().split()
                tour_str = parts[1].strip().split()

                # Parse coordinates: pairs of (x, y) floats
                coords = np.array([float(c) for c in coord_str]).reshape(-1, 2)
                assert coords.shape[0] == num_nodes, (
                    f"Expected {num_nodes} nodes, got {coords.shape[0]}"
                )

                # Parse tour: convert 1-indexed to 0-indexed
                tour = [int(t) - 1 for t in tour_str]

                # Build binary edge labels for the COMPLETE graph.
                # For TSP-50: 50*49/2 = 1225 undirected edges.
                # An edge (i,j) is labeled 1 if it appears in the tour.
                tour_edges = set()
                for idx in range(len(tour) - 1):
                    u, v = tour[idx], tour[idx + 1]
                    tour_edges.add((min(u, v), max(u, v)))
                # Close the loop: last node -> first node
                if len(tour) > 1:
                    u, v = tour[-1], tour[0]
                    tour_edges.add((min(u, v), max(u, v)))

                self.instances.append(coords)
                self.edge_labels.append(tour_edges)

        print(f"Loaded {len(self.instances)} TSP-{num_nodes} instances")

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        coords = self.instances[idx]  # (N, 2)
        tour_edges = self.edge_labels[idx]
        N = self.num_nodes

        # ---------------------------------------------------------
        # Build complete graph
        # ---------------------------------------------------------
        # Edge index: all pairs (i, j) where i < j, then both directions
        src, dst = [], []
        for i in range(N):
            for j in range(i + 1, N):
                src.extend([i, j])
                dst.extend([j, i])
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        # edge_index shape: [2, N*(N-1)] — both directions

        # ---------------------------------------------------------
        # Node features: just the (x, y) coordinates
        # ---------------------------------------------------------
        node_feat = torch.tensor(coords, dtype=torch.float32)  # (N, 2)

        # ---------------------------------------------------------
        # Edge features: Euclidean distance between endpoints
        # ---------------------------------------------------------
        src_coords = coords[edge_index[0]]
        dst_coords = coords[edge_index[1]]
        distances = np.sqrt(((src_coords - dst_coords) ** 2).sum(axis=1, keepdims=True))
        edge_feat = torch.tensor(distances, dtype=torch.float32)  # (num_edges, 1)

        # ---------------------------------------------------------
        # Edge labels: 1 if edge is in tour, 0 otherwise
        # For each directed edge (i,j), label = 1 if (min(i,j), max(i,j)) in tour
        # ---------------------------------------------------------
        labels = []
        for i in range(edge_index.shape[1]):
            u, v = edge_index[0, i].item(), edge_index[1, i].item()
            key = (min(u, v), max(u, v))
            labels.append(1.0 if key in tour_edges else 0.0)
        edge_label = torch.tensor(labels, dtype=torch.float32)

        return node_feat, edge_index, edge_feat, edge_label


# ============================================================
# Part 2: GCN MODEL (Kipf & Welling)
# ============================================================

class GCNLayer(nn.Module):
    """
    One layer of GCN implementing:
        H^{l+1} = σ(D̃^{-1/2} Ã D̃^{-1/2} H^{l} W^{l})

    Step by step:
    1. H @ W        — linear projection (feature transform)
    2. Ã @ (H @ W)  — multiply by adjacency + self-loops (aggregate neighbors)
    3. D̃^{-1/2} normalization on both sides (symmetric normalization)
    4. σ(...)        — apply activation

    For a single node i, this computes:
        h_i^{l+1} = σ( Σ_{j ∈ N(i) ∪ {i}} (1/√(d̃_i · d̃_j)) · h_j^l · W )

    where d̃_i is the degree of node i in the self-loop-augmented graph.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.xavier_uniform_(self.W.weight)

    def forward(self, H, adj_hat):
        """
        Args:
            H:       (N, in_dim) node feature matrix
            adj_hat: (N, N) precomputed D̃^{-1/2} Ã D̃^{-1/2}
        Returns:
            (N, out_dim) updated node features
        """
        # Step 1: Linear projection
        HW = self.W(H)                # (N, out_dim)

        # Step 2+3: Neighbor aggregation with normalization
        out = torch.matmul(adj_hat, HW)  # (N, out_dim)

        return out + self.bias


class GCNForTSP(nn.Module):
    """
    Multi-layer GCN adapted for TSP edge classification.

    Architecture:
        - Stack L GCN layers to get node embeddings
        - For each edge (i,j), concatenate [h_i || h_j || distance(i,j)]
        - Pass through an MLP to predict edge probability

    Note: Standard GCN only produces NODE embeddings.
    To do EDGE classification, we must manually combine node pairs.
    This is a known limitation — GCN was designed for node-level tasks.
    """

    def __init__(self, node_in_dim=2, hidden_dim=64, num_layers=4, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        # GCN layers
        self.layers = nn.ModuleList()
        self.layers.append(GCNLayer(node_in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(GCNLayer(hidden_dim, hidden_dim))
        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])

        # Edge classifier: takes concatenated node pair + distance
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def precompute_adj_hat(self, edge_index, num_nodes):
        """
        Precompute Â = D̃^{-1/2} Ã D̃^{-1/2} from the paper.

        Steps:
            1. Build adjacency matrix A from edge_index
            2. Add self-loops: Ã = A + I_N
            3. Compute degree: D̃_ii = Σ_j Ã_ij
            4. Symmetric normalization: D̃^{-1/2} Ã D̃^{-1/2}
        """
        # Build dense adjacency matrix
        A = torch.zeros(num_nodes, num_nodes, device=edge_index.device)
        A[edge_index[0], edge_index[1]] = 1.0

        # Step 1: Add self-loops  →  Ã = A + I
        A_tilde = A + torch.eye(num_nodes, device=A.device)

        # Step 2: Compute degree matrix  →  D̃
        D_tilde = A_tilde.sum(dim=1)  # (N,)

        # Step 3: D̃^{-1/2}
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(D_tilde + 1e-8))

        # Step 4: Â = D̃^{-1/2} Ã D̃^{-1/2}
        adj_hat = D_inv_sqrt @ A_tilde @ D_inv_sqrt

        return adj_hat

    def forward(self, node_feat, edge_index, edge_feat):
        """
        Args:
            node_feat:  (N, 2) node coordinates
            edge_index: (2, E) edge indices
            edge_feat:  (E, 1) edge distances
        Returns:
            (E,) edge probabilities
        """
        N = node_feat.shape[0]
        adj_hat = self.precompute_adj_hat(edge_index, N)

        # Forward through GCN layers
        h = node_feat
        for i in range(self.num_layers):
            h = self.layers[i](h, adj_hat)
            h = self.norms[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        # h: (N, hidden_dim) — node embeddings

        # Edge classification: concatenate endpoint features + distance
        src_h = h[edge_index[0]]  # (E, hidden_dim)
        dst_h = h[edge_index[1]]  # (E, hidden_dim)
        edge_repr = torch.cat([src_h, dst_h, edge_feat], dim=-1)

        logits = self.edge_mlp(edge_repr).squeeze(-1)  # (E,)
        return logits


# ============================================================
# Part 3: AGNN MODEL (Anisotropic GNN from DIFUSCO)
# ============================================================

class AGNNLayer(nn.Module):
    """
    One layer of Anisotropic GNN as described in DIFUSCO (Section 3.4).

    The key equations are:

        ê_{ij}^{l+1} = P^l e_{ij}^l + Q^l h_i^l + R^l h_j^l
        e_{ij}^{l+1} = e_{ij}^l + MLP_e(BN(ê_{ij}^{l+1}))
        h_i^{l+1}   = h_i^l + α( BN( U^l h_i^l + A_{j∈N_i}(σ(ê_{ij}^{l+1}) ⊙ V^l h_j^l) ) )

    Key differences from GCN:
    ┌─────────────────────┬───────────────────────────────────────────┐
    │ GCN                 │ AGNN                                     │
    ├─────────────────────┼───────────────────────────────────────────┤
    │ Node features only  │ Both node AND edge features              │
    │ Fixed normalization │ Learnable projections (P, Q, R, U, V)    │
    │ Isotropic (treats   │ Anisotropic (direction-aware via gating) │
    │   all neighbors     │                                          │
    │   the same way)     │                                          │
    │ No residual         │ Residual connections on both h and e     │
    └─────────────────────┴───────────────────────────────────────────┘

    The gating mechanism σ(ê) ⊙ Vh means:
    - σ(ê_{ij}) produces a value in [0,1] for each feature dimension
    - This GATES how much of neighbor j's transformed features (V^l h_j)
      flows through edge (i,j) to node i
    - Unlike GCN's fixed 1/√(d_i·d_j) weight, this is LEARNED and
      different for every edge
    """

    def __init__(self, node_dim: int, edge_dim: int):
        super().__init__()
        # Edge update projections
        self.P = nn.Linear(edge_dim, edge_dim, bias=False)   # project current edge feat
        self.Q = nn.Linear(node_dim, edge_dim, bias=False)   # project source node feat
        self.R = nn.Linear(node_dim, edge_dim, bias=False)   # project target node feat

        # Edge MLP with residual
        self.edge_norm = nn.BatchNorm1d(edge_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, edge_dim),
            nn.ReLU(),
            nn.Linear(edge_dim, edge_dim),
        )

        # Node update projections
        self.U = nn.Linear(node_dim, node_dim, bias=False)   # self-transform
        self.V = nn.Linear(node_dim, node_dim, bias=False)   # neighbor transform

        # Node normalization
        self.node_norm = nn.BatchNorm1d(node_dim)

    def forward(self, h, e, edge_index):
        """
        Args:
            h:          (N, node_dim) node features
            e:          (E, edge_dim) edge features
            edge_index: (2, E) edge indices [src, dst]
        Returns:
            h_new: (N, node_dim) updated node features
            e_new: (E, edge_dim) updated edge features
        """
        src, dst = edge_index[0], edge_index[1]
        N = h.shape[0]

        # -------------------------------------------------------
        # Step 1: Edge update
        #   ê_{ij} = P * e_{ij} + Q * h_i + R * h_j
        # -------------------------------------------------------
        e_hat = self.P(e) + self.Q(h[src]) + self.R(h[dst])

        # Residual + MLP:  e_new = e + MLP(BN(ê))
        e_new = e + self.edge_mlp(self.edge_norm(e_hat))

        # -------------------------------------------------------
        # Step 2: Node update (aggregation with gating)
        #
        #   For each node i, aggregate over its neighbors j:
        #     agg_i = SUM_{j ∈ N(i)} [ σ(ê_{ij}) ⊙ V * h_j ]
        #
        #   σ(ê_{ij}) is the GATE — sigmoid squashes to [0,1]
        #   ⊙ is element-wise multiplication (Hadamard product)
        #   This means each edge independently controls the flow
        # -------------------------------------------------------
        gate = torch.sigmoid(e_hat)           # (E, edge_dim) — gate values
        Vh = self.V(h)                         # (N, node_dim) — neighbor transform
        msg = gate * Vh[dst]                   # (E, node_dim) — gated messages

        # Aggregate: sum messages for each target node
        agg = torch.zeros_like(h)             # (N, node_dim)
        agg.index_add_(0, src, msg)           # scatter-add by source node

        # Self-transform + aggregated neighbors, with residual
        Uh = self.U(h)                         # (N, node_dim)
        h_new = h + F.relu(self.node_norm(Uh + agg))

        return h_new, e_new


class AGNNForTSP(nn.Module):
    """
    Full AGNN model for TSP edge classification (DIFUSCO-style).

    Architecture:
        - Initial projection of node coords → node embeddings
        - Initial projection of edge distances → edge embeddings
        - Stack L AGNN layers (updating both node and edge features)
        - Final edge classifier head on edge embeddings

    Unlike GCN, AGNN naturally produces edge embeddings,
    so we don't need the awkward "concatenate node pairs" trick.
    """

    def __init__(self, node_in_dim=2, edge_in_dim=1, hidden_dim=64,
                 num_layers=4, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        # Initial feature projections
        self.node_proj = nn.Linear(node_in_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_in_dim, hidden_dim)

        # AGNN layers
        self.layers = nn.ModuleList([
            AGNNLayer(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        # Edge classification head
        self.edge_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_feat, edge_index, edge_feat):
        """
        Args:
            node_feat:  (N, 2) coordinates
            edge_index: (2, E) edge indices
            edge_feat:  (E, 1) distances
        Returns:
            (E,) edge logits
        """
        # Project raw features to hidden dimension
        h = self.node_proj(node_feat)    # (N, hidden_dim)
        e = self.edge_proj(edge_feat)    # (E, hidden_dim)

        # Forward through AGNN layers
        for layer in self.layers:
            h, e = layer(h, e, edge_index)
            h = F.dropout(h, p=self.dropout, training=self.training)
            e = F.dropout(e, p=self.dropout, training=self.training)

        # Classify edges
        logits = self.edge_head(e).squeeze(-1)  # (E,)
        return logits


# ============================================================
# Part 4: TRAINING & EVALUATION
# ============================================================

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_edges = 0

    for node_feat, edge_index, edge_feat, edge_label in dataloader:
        # Move to device (each item is a single graph)
        node_feat = node_feat.squeeze(0).to(device)
        edge_index = edge_index.squeeze(0).to(device)
        edge_feat = edge_feat.squeeze(0).to(device)
        edge_label = edge_label.squeeze(0).to(device)

        optimizer.zero_grad()

        logits = model(node_feat, edge_index, edge_feat)

        # Weighted BCE loss — tour edges are rare (~50 out of 2450)
        # so we upweight positive examples
        pos_weight = torch.tensor([(edge_label.numel() - edge_label.sum()) / (edge_label.sum() + 1e-8)],
                                  device=device)
        loss = F.binary_cross_entropy_with_logits(logits, edge_label, pos_weight=pos_weight)

        loss.backward()
        optimizer.step()

        # Accuracy
        preds = (torch.sigmoid(logits) > 0.5).float()
        total_correct += (preds == edge_label).sum().item()
        total_edges += edge_label.numel()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_edges
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_edges = 0
    total_tp = 0     # true positives
    total_fp = 0     # false positives
    total_fn = 0     # false negatives

    for node_feat, edge_index, edge_feat, edge_label in dataloader:
        node_feat = node_feat.squeeze(0).to(device)
        edge_index = edge_index.squeeze(0).to(device)
        edge_feat = edge_feat.squeeze(0).to(device)
        edge_label = edge_label.squeeze(0).to(device)

        logits = model(node_feat, edge_index, edge_feat)

        pos_weight = torch.tensor([(edge_label.numel() - edge_label.sum()) / (edge_label.sum() + 1e-8)],
                                  device=device)
        loss = F.binary_cross_entropy_with_logits(logits, edge_label, pos_weight=pos_weight)

        preds = (torch.sigmoid(logits) > 0.5).float()
        total_correct += (preds == edge_label).sum().item()
        total_edges += edge_label.numel()
        total_loss += loss.item()

        # Precision / Recall (important because edges are imbalanced!)
        total_tp += ((preds == 1) & (edge_label == 1)).sum().item()
        total_fp += ((preds == 1) & (edge_label == 0)).sum().item()
        total_fn += ((preds == 0) & (edge_label == 1)).sum().item()

    avg_loss = total_loss / max(len(dataloader), 1)
    accuracy = total_correct / max(total_edges, 1)
    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return avg_loss, accuracy, precision, recall, f1


def run_experiment(model_name, model, train_loader, val_loader, device,
                   lr=1e-3, epochs=50):
    print(f"\n{'='*60}")
    print(f"  Training: {model_name}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'='*60}")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_f1 = 0.0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc, prec, rec, f1 = evaluate(model, val_loader, device)
        scheduler.step()

        elapsed = time.time() - t0

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | "
                  f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}  F1: {f1:.4f} "
                  f"(P={prec:.3f} R={rec:.3f}) | "
                  f"{elapsed:.1f}s")

        if f1 > best_f1:
            best_f1 = f1

    print(f"\n  Best Val F1: {best_f1:.4f}")
    return best_f1


# ============================================================
# Part 5: MAIN
# ============================================================

def create_synthetic_data(num_instances=100, num_nodes=20, file_path="synthetic_tsp.txt"):
    """
    Create synthetic TSP data for quick testing.
    Uses random coordinates and a nearest-neighbor tour as pseudo-ground-truth.
    """
    print(f"Generating {num_instances} synthetic TSP-{num_nodes} instances...")
    lines = []
    for _ in range(num_instances):
        coords = np.random.rand(num_nodes, 2)

        # Nearest-neighbor heuristic for a pseudo-optimal tour
        visited = [0]
        unvisited = set(range(1, num_nodes))
        while unvisited:
            last = visited[-1]
            dists = {j: np.linalg.norm(coords[last] - coords[j]) for j in unvisited}
            nearest = min(dists, key=dists.get)
            visited.append(nearest)
            unvisited.remove(nearest)

        # Format: coords... output tour... (1-indexed)
        coord_str = " ".join(f"{c:.6f}" for c in coords.flatten())
        tour_str = " ".join(str(v + 1) for v in visited)
        lines.append(f"{coord_str} output {tour_str}")

    with open(file_path, "w") as f:
        f.write("\n".join(lines))

    print(f"  Saved to {file_path}")
    return file_path


def main():
    parser = argparse.ArgumentParser(description="GCN vs AGNN for TSP")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to TSP data file. If not provided, generates synthetic data.")
    parser.add_argument("--num_nodes", type=int, default=20,
                        help="Number of nodes per TSP instance (default: 20)")
    parser.add_argument("--hidden_dim", type=int, default=64,
                        help="Hidden dimension for both models")
    parser.add_argument("--num_layers", type=int, default=4,
                        help="Number of GNN layers")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--synthetic_instances", type=int, default=200,
                        help="Number of synthetic instances to generate (if no data_path)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load or generate data
    if args.data_path:
        file_path = args.data_path
    else:
        file_path = create_synthetic_data(
            num_instances=args.synthetic_instances,
            num_nodes=args.num_nodes,
        )

    dataset = TSPDataset(file_path, num_nodes=args.num_nodes)

    # Train/val split (80/20)
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    # batch_size=1 because each graph has different topology in principle
    # (though for complete graphs they're all the same structure)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    # -------------------------------------------------------
    # Model 1: GCN
    # -------------------------------------------------------
    gcn_model = GCNForTSP(
        node_in_dim=2,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    )
    gcn_f1 = run_experiment("GCN (Kipf & Welling)", gcn_model,
                            train_loader, val_loader, device,
                            lr=args.lr, epochs=args.epochs)

    # -------------------------------------------------------
    # Model 2: AGNN (DIFUSCO-style)
    # -------------------------------------------------------
    agnn_model = AGNNForTSP(
        node_in_dim=2,
        edge_in_dim=1,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    )
    agnn_f1 = run_experiment("AGNN (DIFUSCO-style)", agnn_model,
                             train_loader, val_loader, device,
                             lr=args.lr, epochs=args.epochs)

    # -------------------------------------------------------
    # Summary
    # -------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  GCN  Best Val F1: {gcn_f1:.4f}")
    print(f"  AGNN Best Val F1: {agnn_f1:.4f}")
    print(f"{'='*60}")

    if agnn_f1 > gcn_f1:
        print("  → AGNN outperforms GCN (expected!)")
        print("    AGNN's edge features + gating mechanism provide richer")
        print("    representations for edge-level tasks like TSP.")
    else:
        print("  → GCN performed competitively (possible with small data/epochs)")
        print("    With more data and training, AGNN typically pulls ahead.")


if __name__ == "__main__":
    main()
