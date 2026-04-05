import argparse
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from src.agnn_layer import AGNNForTSP
from src.dataset import TSPDataset
from src.gcn_layer import GCNForTSP
from src.train import evaluate, train_one_epoch


def run_experiment(
    model_name,
    model,
    train_loader,
    val_loader,
    device,
    lr=1e-3,
    epochs=50,
    use_wandb=False,
    wandb_config=None,
):
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'=' * 60}")
    print(f"  Training: {model_name}")
    print(f"  Parameters: {num_params:,}")
    print(f"{'=' * 60}")

    if use_wandb:
        run_config = {**(wandb_config or {}), "model": model_name, "params": num_params}
        wandb.init(
            project=run_config.pop("wandb_project", "gcn-vs-agnn-tsp"),
            name=model_name,
            config=run_config,
            reinit=True,
        )
        wandb.watch(model, log="gradients", log_freq=10)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_f1 = 0.0
    epoch_bar = tqdm(range(1, epochs + 1), desc=f"  {model_name}", unit="epoch")
    for epoch in epoch_bar:
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc, prec, rec, f1 = evaluate(model, val_loader, device)
        scheduler.step()

        elapsed = time.time() - t0

        epoch_bar.set_postfix(
            loss=f"{train_loss:.4f}",
            acc=f"{train_acc:.4f}",
            val_f1=f"{f1:.4f}",
            t=f"{elapsed:.1f}s",
        )

        if use_wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/accuracy": train_acc,
                    "val/loss": val_loss,
                    "val/accuracy": val_acc,
                    "val/precision": prec,
                    "val/recall": rec,
                    "val/f1": f1,
                    "lr": scheduler.get_last_lr()[0],
                    "epoch_time_s": elapsed,
                },
            )

        if f1 > best_f1:
            best_f1 = f1

    print(f"\n  Best Val F1: {best_f1:.4f}")

    if use_wandb:
        wandb.summary["best_val_f1"] = best_f1
        wandb.finish()

    return best_f1


def main():
    parser = argparse.ArgumentParser(description="GCN vs AGNN for TSP")
    parser.add_argument(
        "--data_path",
        type=str,
        default="src/TSP-50nodes-100instances.txt",
        help="Path to TSP data file.",
    )
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=50,
        help="Number of nodes per TSP instance (default: 50)",
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=64, help="Hidden dimension for both models"
    )
    parser.add_argument(
        "--num_layers", type=int, default=4, help="Number of GNN layers"
    )
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="gcn-vs-agnn-tsp",
        help="wandb project name",
    )
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    dataset = TSPDataset(args.data_path, num_nodes=args.num_nodes)

    # Train/val split (80/20)
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    # -------------------------------------------------------
    # Model 1: GCN
    # -------------------------------------------------------
    gcn_model = GCNForTSP(
        node_in_dim=2,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=0.1,
    )
    wandb_config = {
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "epochs": args.epochs,
        "lr": args.lr,
        "num_nodes": args.num_nodes,
        "wandb_project": args.wandb_project,
    }

    gcn_f1 = run_experiment(
        "GCN (Kipf & Welling)",
        gcn_model,
        train_loader,
        val_loader,
        device,
        lr=args.lr,
        epochs=args.epochs,
        use_wandb=args.wandb,
        wandb_config=wandb_config,
    )

    # -------------------------------------------------------
    # Model 2: AGNN (DIFUSCO-style)
    # -------------------------------------------------------
    agnn_model = AGNNForTSP(
        node_in_dim=2,
        edge_in_dim=1,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=0.1,
    )
    agnn_f1 = run_experiment(
        "AGNN (DIFUSCO-style)",
        agnn_model,
        train_loader,
        val_loader,
        device,
        lr=args.lr,
        epochs=args.epochs,
        use_wandb=args.wandb,
        wandb_config=wandb_config,
    )

    # -------------------------------------------------------
    # Summary
    # -------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("  RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(f"  GCN  Best Val F1: {gcn_f1:.4f}")
    print(f"  AGNN Best Val F1: {agnn_f1:.4f}")
    print(f"{'=' * 60}")

    if agnn_f1 > gcn_f1:
        print("  → AGNN outperforms GCN (expected!)")
        print("    AGNN's edge features + gating mechanism provide richer")
        print("    representations for edge-level tasks like TSP.")
    else:
        print("  → GCN performed competitively (possible with small data/epochs)")
        print("    With more data and training, AGNN typically pulls ahead.")


# uv run python -m src.run
if __name__ == "__main__":
    main()
