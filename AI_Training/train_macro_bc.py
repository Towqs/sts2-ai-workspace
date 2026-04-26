import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class MacroBCModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def train(processed_dir):
    processed_dir = Path(processed_dir)
    x_path = processed_dir / "X_train.npy"
    y_path = processed_dir / "Y_train.npy"
    vocab_path = processed_dir / "vocab.json"
    metadata_path = processed_dir / "metadata.json"

    if not x_path.exists() or not y_path.exists() or not vocab_path.exists():
        print("Macro data files not found. Please run macro_data_pipeline.py first.")
        return 1

    print("Loading macro dataset...")
    x_train = np.load(x_path)
    y_train = np.load(y_path)
    vocab = load_json(vocab_path, {})
    metadata = load_json(metadata_path, {})

    if len(y_train) == 0:
        print("No macro samples available. Training skipped.")
        return 0

    num_actions = len(vocab.get("actions", {}))
    input_dim = int(x_train.shape[1])
    if num_actions <= 2:
        print("Macro action space is empty. Training skipped.")
        return 0

    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(7)

    print(f"X_train shape: {x_train.shape}")
    print(f"Y_train shape: {y_train.shape}")
    print(f"Input Features: {input_dim}, Action Space: {num_actions}")
    print(f"Metadata samples: {metadata.get('samples', len(y_train))}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    y_train = np.clip(y_train, 0, num_actions - 1)
    x_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.int64)

    batch_size = min(32, max(1, len(y_tensor)))
    dataset = TensorDataset(x_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MacroBCModel(input_dim, num_actions).to(device)

    counts = np.bincount(y_train, minlength=num_actions).astype(np.float32)
    weights = np.zeros(num_actions, dtype=np.float32)
    present = counts > 0
    weights[present] = 1.0 / np.sqrt(counts[present])
    if weights[present].sum() > 0:
        weights[present] *= present.sum() / weights[present].sum()
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 80 if len(y_train) < 100 else 120
    best_loss = float("inf")
    best_acc = 0.0
    best_path = processed_dir / "macro_bc_model_best.pth"

    print("\nStarting Macro Training (Behavioral Cloning)...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predicted = outputs.argmax(dim=1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        avg_loss = total_loss / max(len(dataloader), 1)
        acc = correct / max(total, 1) * 100.0
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_acc = acc
            torch.save(model.state_dict(), best_path)

    summary = {
        "samples": int(len(y_train)),
        "features": input_dim,
        "actions": num_actions,
        "batch_size": batch_size,
        "epochs": epochs,
        "best_loss": float(best_loss),
        "best_train_accuracy": float(best_acc),
        "device": str(device),
        "model_path": str(best_path.name),
    }
    save_json(processed_dir / "training_summary.json", summary)

    print(f"Training complete. Best macro model saved to {best_path.name} (Loss: {best_loss:.4f}, Acc: {best_acc:.2f}%)")
    return 0


if __name__ == "__main__":
    workspace_dir = Path(__file__).resolve().parents[1]
    processed = workspace_dir / "AI_Training" / "ProcessedMacroParams"
    raise SystemExit(train(processed))
