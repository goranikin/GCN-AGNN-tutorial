# GCN vs AGNN for TSP Edge Classification

Comparing two Graph Neural Network architectures on the Travelling Salesman Problem (TSP): a standard **GCN** (Kipf & Welling, 2017) and an **Anisotropic GNN** (AGNN) inspired by [DIFUSCO](https://arxiv.org/abs/2302.08224).

Both models are trained to classify edges — predicting which edges belong to the optimal tour.

## Why AGNN over GCN?

| | GCN | AGNN |
|---|---|---|
| Message passing | Isotropic (same weight for all neighbors) | Anisotropic (learned gating per edge) |
| Edge features | Not natively supported; requires post-hoc concatenation of node pairs | First-class edge embeddings updated at every layer |
| Edge prediction | Concatenate `[src_h, dst_h, edge_feat]` then MLP | Directly classify from edge embeddings |
| Expressiveness | Limited by symmetric normalization | Richer representations via `P`, `Q`, `R` projections + sigmoid gating |

TSP is fundamentally an **edge-level** task, which makes AGNN's native edge representations a natural fit.

## Project Structure

```
src/
├── dataset.py          # TSPDataset — parses instances, builds fully-connected graphs
├── gcn_layer.py        # GCNLayer + GCNForTSP (spectral convolution → edge MLP)
├── agnn_layer.py       # AGNNLayer + AGNNForTSP (DIFUSCO-style anisotropic message passing)
├── train.py            # train_one_epoch() and evaluate() with F1/precision/recall
├── run.py              # CLI entry point — trains both models and compares results
└── TSP-50nodes-100instances.txt   # 100 TSP-50 instances with optimal tours
```

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Usage

### Basic run

```bash
uv run python -m src.run
```

### With wandb logging

```bash
uv run python -m src.run --wandb
```

### All options

```bash
uv run python -m src.run \
  --data_path src/TSP-50nodes-100instances.txt \
  --num_nodes 50 \
  --hidden_dim 64 \
  --num_layers 4 \
  --epochs 50 \
  --lr 1e-3 \
  --wandb \
  --wandb_project gcn-vs-agnn-tsp
```

| Flag | Default | Description |
|---|---|---|
| `--data_path` | `src/TSP-50nodes-100instances.txt` | Path to TSP data file |
| `--num_nodes` | `50` | Nodes per TSP instance |
| `--hidden_dim` | `64` | Hidden dimension for both models |
| `--num_layers` | `4` | Number of GNN layers |
| `--epochs` | `50` | Training epochs |
| `--lr` | `1e-3` | Learning rate |
| `--wandb` | off | Enable Weights & Biases logging |
| `--wandb_project` | `gcn-vs-agnn-tsp` | wandb project name |

## Device Support

Automatically selects the best available device:

- **CUDA** (NVIDIA GPUs)
- **MPS** (Apple Silicon — M1/M2/M3/M4)
- **CPU** (fallback)

## Data Format

Each line in the data file contains one TSP instance:

```
x1 y1 x2 y2 ... xN yN output n1 n2 ... nN n1
```

- Coordinates are pairs of floats in `[0, 1]`
- Tour indices are 1-based, forming a closed loop

## Key Equations

### GCN Layer

\[H^{(l+1)} = \sigma\!\Big(\hat{D}^{-1/2}\,\hat{A}\,\hat{D}^{-1/2}\,H^{(l)}\,W^{(l)}\Big)\]

where \(\hat{A} = A + I\) (self-loops) and \(\hat{D}\) is the degree matrix of \(\hat{A}\).

### AGNN Layer (DIFUSCO)

**Edge update:**
\[\hat{e}_{ij}^{(l+1)} = P^{(l)} e_{ij}^{(l)} + Q^{(l)} h_i^{(l)} + R^{(l)} h_j^{(l)}\]
\[e_{ij}^{(l+1)} = e_{ij}^{(l)} + \text{MLP}_e\!\Big(\text{BN}\big(\hat{e}_{ij}^{(l+1)}\big)\Big)\]

**Node update:**
\[h_i^{(l+1)} = h_i^{(l)} + \alpha\,\text{BN}\!\Big(U^{(l)} h_i^{(l)} + \sum_{j \in \mathcal{N}(i)} \sigma\!\big(\hat{e}_{ij}^{(l+1)}\big) \odot V^{(l)} h_j^{(l)}\Big)\]

## References

- Kipf & Welling (2017). [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)
- Sun et al. (2023). [DIFUSCO: Graph-based Diffusion Solvers for Combinatorial Optimization](https://arxiv.org/abs/2302.08224)
