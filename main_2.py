import os
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from copy import deepcopy
from dataset import TrainDataset, TestDataset
from model import OptimizedLSTMModel  # Ensure the optimized model is imported from model.py

# Device selection
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Define the training and validation functions
def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device):
    model.train()
    train_loss = 0.0
    for seq, cdd, _, labels in tqdm(train_loader):
        seq, cdd, labels = seq.to(device), cdd.to(device), labels.to(device)
        
        optimizer.zero_grad()
        preds = model(seq)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    scheduler.step()  # Step the learning rate scheduler after each epoch
    return train_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct_order_count = 0
    with torch.no_grad():
        for seq, cdd, _, labels in val_loader:
            seq, cdd, labels = seq.to(device), cdd.to(device), labels.to(device)
            
            preds = model(seq)
            loss = criterion(preds, labels)
            val_loss += loss.item()
            
            # Order correctness evaluation
            pred_order = torch.argsort(preds, dim=1, descending=True)
            correct_order_count += (pred_order == labels).all(dim=1).sum().item()

    accuracy = correct_order_count / len(val_loader.dataset)
    return val_loss / len(val_loader), accuracy

# Main training process
if __name__ == "__main__":
    # Hyperparameters
    batch_size = 128
    n_epochs = 20
    learning_rate = 0.001
    weight_decay = 1e-5

    # Seed and dataset setup
    seed = 123
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_set = TrainDataset()
    train_size = int(0.8 * len(train_set))
    val_size = len(train_set) - train_size
    train_data, val_data = random_split(train_set, [train_size, val_size], torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    # Model, optimizer, and loss function
    model = OptimizedLSTMModel(input_size=1280, hidden_size=2560, n_layers=2, dropout=0.3).to(device)
    criterion = torch.nn.CrossEntropyLoss()  # Adjusted based on your requirements
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # Decrease LR every 10 epochs

    # Track best model
    best_loss = np.inf
    best_acc = 0.0
    best_model = deepcopy(model.state_dict())

    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1}/{n_epochs}:")
        
        # Training
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        print(f"Train loss: {train_loss}")

        # Validation
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"Val loss: {val_loss}, Val accuracy: {val_acc}")

        # Save best model based on validation accuracy
        if val_acc > best_acc:
            best_acc = val_acc
            best_model = deepcopy(model.state_dict())

    # Save the best model
    os.makedirs("trained_models", exist_ok=True)
    model.load_state_dict(best_model)
    torch.save(model.state_dict(), "trained_models/optimized_model.pth")
    print(f"Best model saved with accuracy: {best_acc}")