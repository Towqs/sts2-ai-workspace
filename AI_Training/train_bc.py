import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import time

class CombatBCModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CombatBCModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

def train(processed_dir):
    x_path = os.path.join(processed_dir, 'X_train.npy')
    y_path = os.path.join(processed_dir, 'Y_train.npy')
    vocab_path = os.path.join(processed_dir, 'vocab.json')
    metadata_path = os.path.join(processed_dir, 'metadata.json')
    
    if not os.path.exists(x_path):
        print("Data files not found. Please run data_pipeline.py first.")
        return
        
    # Load data
    print("Loading datasets...")
    X_train = np.load(x_path)
    Y_train = np.load(y_path)
    
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    
    num_actions = len(vocab['actions'])
    input_dim = X_train.shape[1]
    
    print(f"X_train shape: {X_train.shape}")
    print(f"Y_train shape: {Y_train.shape}")
    print(f"Input Features: {input_dim}, Action Space: {num_actions}")
    if len(Y_train) == 0:
        print("No training samples available. Check collection settings and run quality filters.")
        return
    
    # Check GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create DataLoader
    X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    Y_tensor = torch.tensor(Y_train, dtype=torch.int64).to(device)
    
    # Quick fix for any out of bounds action IDs (UNKNOWN=1)
    Y_tensor = torch.clamp(Y_tensor, 0, num_actions - 1)
    
    dataset = TensorDataset(X_tensor, Y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Init Model
    model = CombatBCModel(input_dim, num_actions).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("\nStarting Training (Behavioral Cloning)...")
    epochs = 100
    best_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
        avg_loss = total_loss / len(dataloader)
        acc = correct / total * 100
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")
            
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(processed_dir, 'bc_model_best.pth'))
            
    print(f"Training complete. Best model saved to bc_model_best.pth (Loss: {best_loss:.4f})")
    summary = {
        "samples": int(len(Y_train)),
        "features": int(input_dim),
        "actions": int(num_actions),
        "epochs": int(epochs),
        "best_loss": float(best_loss),
        "device": str(device),
        "model_path": "bc_model_best.pth",
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PROCESSED_DIR = os.path.join(WORKSPACE_DIR, "AI_Training", "ProcessedParams")
    train(PROCESSED_DIR)
