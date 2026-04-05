import torch
import torch.nn.functional as F
from tqdm import tqdm


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_edges = 0

    for node_feat, edge_index, edge_feat, edge_label in tqdm(
        dataloader, desc="    Train", leave=False, unit="batch"
    ):
        node_feat = node_feat.squeeze(0).to(device)
        edge_index = edge_index.squeeze(0).to(device)
        edge_feat = edge_feat.squeeze(0).to(device)
        edge_label = edge_label.squeeze(0).to(device)

        optimizer.zero_grad()

        logits = model(node_feat, edge_index, edge_feat)

        pos_weight = torch.tensor(
            [(edge_label.numel() - edge_label.sum()) / (edge_label.sum() + 1e-8)],
            device=device,
        )
        loss = F.binary_cross_entropy_with_logits(
            logits, edge_label, pos_weight=pos_weight
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

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
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for node_feat, edge_index, edge_feat, edge_label in tqdm(
        dataloader, desc="    Val  ", leave=False, unit="batch"
    ):
        node_feat = node_feat.squeeze(0).to(device)
        edge_index = edge_index.squeeze(0).to(device)
        edge_feat = edge_feat.squeeze(0).to(device)
        edge_label = edge_label.squeeze(0).to(device)

        logits = model(node_feat, edge_index, edge_feat)

        pos_weight = torch.tensor(
            [(edge_label.numel() - edge_label.sum()) / (edge_label.sum() + 1e-8)],
            device=device,
        )
        loss = F.binary_cross_entropy_with_logits(
            logits, edge_label, pos_weight=pos_weight
        )

        preds = torch.sigmoid(logits) > 0.5
        labels = edge_label.bool()
        total_correct += (preds == labels).sum().item()
        total_edges += edge_label.numel()
        total_loss += loss.item()
        total_tp += (preds & labels).sum().item()
        total_fp += (preds & ~labels).sum().item()
        total_fn += (~preds & labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_edges
    precision = total_tp / (total_tp + total_fp + 1e-8)
    recall = total_tp / (total_tp + total_fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return avg_loss, accuracy, precision, recall, f1
