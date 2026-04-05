import torch
import torch.nn as nn
import torch.nn.functional as F


class AGNNLayer(nn.Module):
    """
    One layer of Anisotropic GNN as described in DIFUSCO (Section 3.4).

    The key equations are:

        ê_{ij}^{l+1} = P^l e_{ij}^l + Q^l h_i^l + R^l h_j^l
        e_{ij}^{l+1} = e_{ij}^l + MLP_e(BN(ê_{ij}^{l+1}))
        h_i^{l+1}   = h_i^l + α( BN( U^l h_i^l + A_{j∈N_i}(σ(ê_{ij}^{l+1}) ⊙ V^l h_j^l) ) )
    """

    def __init__(self, node_dim: int, edge_dim: int):
        super().__init__()
        self.P = nn.Linear(edge_dim, edge_dim, bias=False)
        self.Q = nn.Linear(node_dim, edge_dim, bias=False)
        self.R = nn.Linear(node_dim, edge_dim, bias=False)

        self.edge_norm = nn.BatchNorm1d(edge_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, edge_dim),
            nn.ReLU(),
            nn.Linear(edge_dim, edge_dim),
        )

        self.U = nn.Linear(node_dim, node_dim, bias=False)
        self.V = nn.Linear(node_dim, node_dim, bias=False)

        self.node_norm = nn.BatchNorm1d(node_dim)

    def forward(self, h, e, edge_index):
        src, dst = edge_index[0], edge_index[1]

        # Step 1: Edge update
        #   ê_{ij} = P * e_{ij} + Q * h_i + R * h_j
        e_hat = self.P(e) + self.Q(h[src]) + self.R(h[dst])

        # Residual + MLP:  e_new = e + MLP(BN(ê))
        e_new = e + self.edge_mlp(self.edge_norm(e_hat))

        # Step 2: Node update (aggregation with gating)
        #   For each node i, aggregate over its neighbors j:
        #     agg_i = SUM_{j ∈ N(i)} [ σ(ê_{ij}) ⊙ V * h_j ]
        #
        #   σ(ê_{ij}) is the GATE — sigmoid squashes to [0,1]
        #   ⊙ is element-wise multiplication (Hadamard product)
        #   This means each edge independently controls the flow
        gate = torch.sigmoid(e_hat)
        Vh = self.V(h)
        msg = gate * Vh[dst]

        # Aggregate: sum messages for each target node
        agg = torch.zeros_like(h)
        agg.index_add_(0, src, msg)

        # Self-transform + aggregated neighbors, with residual
        Uh = self.U(h)
        h_new = h + self.alpha * self.node_norm(Uh + agg)

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

    def __init__(
        self, node_in_dim=2, edge_in_dim=1, hidden_dim=64, num_layers=4, dropout=0.1
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        # Initial feature projections
        self.node_proj = nn.Linear(node_in_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_in_dim, hidden_dim)

        # AGNN layers
        self.layers = nn.ModuleList(
            [AGNNLayer(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )

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
        h = self.node_proj(node_feat)  # (N, hidden_dim)
        e = self.edge_proj(edge_feat)  # (E, hidden_dim)

        # Forward through AGNN layers
        for layer in self.layers:
            h, e = layer(h, e, edge_index)
            h = F.dropout(h, p=self.dropout, training=self.training)
            e = F.dropout(e, p=self.dropout, training=self.training)

        # Classify edges
        logits = self.edge_head(e).squeeze(-1)  # (E,)
        return logits
