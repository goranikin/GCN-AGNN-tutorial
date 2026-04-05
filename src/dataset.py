from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class TSPDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        num_nodes: int,
    ):
        self.num_nodes: int = num_nodes
        self.instance_list: list[np.ndarray[tuple[int, int], np.dtype[Any]]] = []
        self.edge_label_list: list[set[tuple[int, int]]] = []

        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split("output")
                coord_str: list[str] = parts[0].strip().split()
                tour_str: list[str] = parts[1].strip().split()

                coords: np.ndarray[tuple[int, int], np.dtype[Any]] = np.array(
                    [float(c) for c in coord_str]
                ).reshape(-1, 2)
                assert coords.shape[0] == num_nodes, (
                    f"Expected {num_nodes} coordinates, got {coords.shape[0]}"
                )

                tour: list[int] = [int(t) - 1 for t in tour_str]

                tour_edges: set[tuple[int, int]] = set()
                for idx in range(len(tour) - 1):
                    u, v = tour[idx], tour[idx + 1]
                    tour_edges.add((min(u, v), max(u, v)))

                if len(tour) > 1:
                    u, v = tour[-1], tour[0]
                    tour_edges.add((min(u, v), max(u, v)))

                self.instance_list.append(coords)
                self.edge_label_list.append(tour_edges)

    def __len__(self):
        return len(self.instance_list)

    def __getitem__(self, index):
        coords: np.ndarray[tuple[int, int], np.dtype[Any]] = self.instance_list[index]
        tour_edges: set[tuple[int, int]] = self.edge_label_list[index]
        N: int = self.num_nodes

        src: list[int] = []
        dst: list[int] = []
        for i in range(N):
            for j in range(i + 1, N):
                src.extend([i, j])
                dst.extend([j, i])

        # [2, num_edges]
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        # [num_nodes, 2]: x, y coordinates
        node_features = torch.tensor(coords, dtype=torch.float)

        src_coords = node_features[edge_index[0]]
        dst_coords = node_features[edge_index[1]]
        distances = torch.norm(src_coords - dst_coords, dim=1)
        edge_features = distances

        labels = []
        for i in range(edge_index.shape[1]):
            u, v = edge_index[0, i].item(), edge_index[1, i].item()
            key = (min(u, v), max(u, v))
            labels.append(1.0 if key in tour_edges else 0.0)
        edge_label = torch.tensor(labels, dtype=torch.float)

        return node_features, edge_index, edge_features, edge_label


def test():
    import os

    NUM_NODES = 50
    dataset = TSPDataset(
        os.path.join(os.path.dirname(__file__), "TSP-50nodes-100instances.txt"),
        num_nodes=NUM_NODES,
    )

    node_features, edge_index, edge_features, edge_label = dataset[0]
    print(node_features.shape)
    print(edge_index.shape)
    print(edge_features.shape)
    print(edge_label.shape)


if __name__ == "__main__":
    test()
