import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.xavier_uniform_(self.W.weight)

    def forward(self, H, adj_hat):
        """
        Agrs:
          H: (N, in_dim) node feature matrix
          adj_hat: (N, N) normalized adjacency matrix
        Returns:
          (N, out_dim) node feature matrix
        """

        HW = self.W(H)
        out = torch.matmul(adj_hat, HW) + self.bias
        return out


class GCNForTSP(nn.Module):
    def __init__(self, node_in_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.layers = nn.ModuleList()
        self.layers.append(GCNLayer(node_in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(GCNLayer(hidden_dim, hidden_dim))

        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.norms.append(nn.LayerNorm(hidden_dim))

        # (hidden_dim * 2 + 1, 1) = (edge_repr_dim, 1)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def precompute_adj_hat(self, edge_index, num_nodes):
        """
        Â = D̃^{-1/2} Ã D̃^{-1/2}
        """
        A = torch.zeros(num_nodes, num_nodes, device=edge_index.device)
        A[edge_index[0], edge_index[1]] = 1.0

        # (N, N)
        A_tilde = A + torch.eye(num_nodes, device=edge_index.device)
        D_tilde = A_tilde.sum(dim=1)
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(D_tilde + 1e-8))
        adj_hat = torch.matmul(torch.matmul(D_inv_sqrt, A_tilde), D_inv_sqrt)

        return adj_hat

    def forward(self, node_features, edge_index, edge_features):
        """
        Args:
          node_features: (N, node_in_dim) node feature matrix
          edge_index: (2, num_edges) edge index matrix
          edge_features: (num_edges, edge_in_dim) edge feature matrix
        Returns:
          (num_edges, 1) edge label matrix
        """
        N = node_features.shape[0]
        # Cache adj_hat (only recompute if graph size changes)
        if not hasattr(self, "_cached_adj") or self._cached_N != N:
            self._cached_adj = self.precompute_adj_hat(edge_index, N)
            self._cached_N = N
        adj_hat = self._cached_adj

        # (N, node_in_dim)
        h = node_features
        for layer, norm in zip(self.layers, self.norms):
            # (N, node_in_dim) -> (N, hidden_dim) when first layer
            # (N, hidden_dim) -> (N, hidden_dim) otherwise
            h = layer(h, adj_hat)
            # (N, hidden_dim)
            h = norm(h)
            # (N, hidden_dim)
            h = F.relu(h)
            # (N, hidden_dim)
            h = F.dropout(h, self.dropout, training=self.training)

        # (num_edges, hidden_dim)
        src_h = h[edge_index[0]]
        dst_h = h[edge_index[1]]

        if edge_features.dim() == 1:
            edge_features = edge_features.unsqueeze(-1)

        # (num_edges, hidden_dim * 2 + 1)
        edge_repr = torch.cat([src_h, dst_h, edge_features], dim=1)

        # (num_edges, 1)
        logits = self.edge_mlp(edge_repr).squeeze(-1)
        return logits
