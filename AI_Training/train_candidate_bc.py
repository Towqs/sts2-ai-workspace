import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class CandidateBCScorer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def group_top1_accuracy(logits, labels, groups):
    if len(labels) == 0:
        return 0.0
    total = 0
    correct = 0
    unique_groups = np.unique(groups)
    logits = np.asarray(logits)
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    for group_id in unique_groups:
        mask = groups == group_id
        if not np.any(mask):
            continue
        group_labels = labels[mask]
        if group_labels.max() <= 0:
            continue
        group_logits = logits[mask]
        total += 1
        correct += int(group_labels[int(group_logits.argmax())] == 1)
    return (correct / total * 100.0) if total else 0.0


def train(processed_dir):
    x_path = os.path.join(processed_dir, "candidate_X_train.npy")
    y_path = os.path.join(processed_dir, "candidate_Y_train.npy")
    group_path = os.path.join(processed_dir, "candidate_group_train.npy")
    metadata_path = os.path.join(processed_dir, "candidate_metadata.json")

    if not os.path.exists(x_path) or not os.path.exists(y_path) or not os.path.exists(group_path):
        print("Candidate data files not found. Run data_pipeline.py first.")
        return

    print("Loading candidate datasets...")
    x_train = np.load(x_path)
    y_train = np.load(y_path)
    groups = np.load(group_path)

    if len(y_train) == 0:
        print("No candidate rows available. Check combat action enumeration and data filters.")
        return

    input_dim = int(x_train.shape[1])
    positives = int(y_train.sum())
    negatives = int(len(y_train) - positives)
    if positives == 0:
        print("No positive candidate labels available; cannot train scorer.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Candidate X shape: {x_train.shape}")
    print(f"Candidate positives: {positives}, negatives: {negatives}, groups: {len(np.unique(groups))}")
    print(f"Using device: {device}")

    x_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)
    group_array = np.asarray(groups)
    dataset = TensorDataset(x_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    model = CandidateBCScorer(input_dim).to(device)
    pos_weight = torch.tensor([max(1.0, negatives / max(positives, 1))], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 80
    best_loss = float("inf")
    best_top1 = 0.0
    started = time.time()

    print("\nStarting candidate-action BC scorer training...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(dataloader), 1)

        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                logits = model(x_tensor.to(device)).detach().cpu().numpy()
            top1 = group_top1_accuracy(logits, y_train, group_array)
            print(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | Group Top1: {top1:.2f}%")
        else:
            top1 = best_top1

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_top1 = max(best_top1, top1)
            torch.save(model.state_dict(), os.path.join(processed_dir, "candidate_bc_model_best.pth"))

    model.eval()
    with torch.no_grad():
        final_logits = model(x_tensor.to(device)).detach().cpu().numpy()
    best_top1 = max(best_top1, group_top1_accuracy(final_logits, y_train, group_array))

    summary = {
        "samples": int(len(y_train)),
        "groups": int(len(np.unique(groups))),
        "positives": int(positives),
        "negatives": int(negatives),
        "features": int(input_dim),
        "epochs": int(epochs),
        "best_loss": float(best_loss),
        "best_group_top1": float(best_top1),
        "device": str(device),
        "elapsed_sec": round(time.time() - started, 2),
        "model_path": "candidate_bc_model_best.pth",
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Training complete. Best candidate scorer saved to candidate_bc_model_best.pth (Loss: {best_loss:.4f})")


if __name__ == "__main__":
    WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PROCESSED_DIR = os.path.join(WORKSPACE_DIR, "AI_Training", "ProcessedParams")
    train(PROCESSED_DIR)
