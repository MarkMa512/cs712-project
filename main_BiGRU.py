import os
import random
import zipfile
from copy import deepcopy
from time import time

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset_own import TestDataset, TrainDataset
from model import BiGRUModel


def seed_everything(TORCH_SEED):
    random.seed(TORCH_SEED)
    os.environ["PYTHONHASHSEED"] = str(TORCH_SEED)
    np.random.seed(TORCH_SEED)
    torch.manual_seed(TORCH_SEED)
    torch.cuda.manual_seed_all(TORCH_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    batch_size = 256
    n_epochs = 10
    seed = 123
    dropout=0.001

    seed_everything(seed)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    # train/val split
    train_set = TrainDataset()
    train_size = int(0.8 * len(train_set))
    val_size = len(train_set) - train_size
    train_data, val_data = random_split(train_set, [train_size, val_size], torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    # model = BiGRUModel().to(device) # 0.26
    # model = BiGRUModel(input_size=1280, hidden_size=1280, n_layers=2, batch_size=128).to(device) 
    model = BiGRUModel(input_size=1280, hidden_size=1280, n_layers=1, batch_size=256).to(device) 


    model_name = 'BiGRUModel'

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_loss = np.inf
    best_acc = 0.0
    best_model = deepcopy(model.state_dict())

    for epoch in range(n_epochs):
        # training
        start = time()
        model.train()
        print(f"Epoch {epoch+1}/{n_epochs}:")
        train_loss = 0.0

        for seq, cdd, next_point, labels in tqdm(train_loader):
            seq = seq.to(device)
            cdd = cdd.to(device)
            labels = labels.to(device)
            model.zero_grad()
            preds = model(seq, cdd)

            loss = criterion(preds, labels)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Train loss: {train_loss/train_size}")
        print(f"Training time: {time()-start}")

        # validation
        start = time()
        model.eval()
        val_loss = 0.0
        val_hit = 0

        for seq, cdd, next_point, labels in tqdm(val_loader):
            seq = seq.to(device)
            cdd = cdd.to(device)
            labels = labels.to(device)
            preds = model(seq, cdd)
            targets = torch.ones(len(seq)).to(device)
            loss = criterion(preds, labels)
            val_loss += loss.item()
            batch_output = torch.argsort(torch.argsort(preds, dim=1, descending=True), dim=1).cpu().numpy()
            batch_labels = np.array([[0, 1, 2] for _ in range(len(batch_output))])
            val_hit += sum(not l.any() for l in (batch_labels - batch_output))

        val_acc = val_hit / len(val_data)
        print(f"Val accuracy: {val_acc}")
        print(f"Eval time: {time()-start}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_model = deepcopy(model.state_dict())

    model.load_state_dict(best_model)
    # Ensure the 'trained_models' directory exists
    os.makedirs("trained_models", exist_ok=True)
    torch.save(model.state_dict(), f"trained_models/model_{model_name}.pth")

    ### generate predictions for public test
    test_set = TestDataset()
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    model.eval()
    start = time()

    # predictions
    pred_labels = []
    next_points = []
    for seq, cdd in tqdm(test_loader):
        seq = seq.to(device)
        cdd = cdd.to(device)
        preds, hidden = model(seq, cdd, return_hidden=True)
        batch_output = torch.argsort(torch.argsort(preds, dim=1, descending=True), dim=1)
        pred_labels.extend(batch_output.cpu().tolist())
        next_points.extend(hidden.cpu().tolist())

    combined_preds = np.concatenate([np.array(pred_labels), np.array(next_points)], axis=1)
    # save predictions
    file_name = f"gru_mse_{model_name}"
    np.savetxt(file_name + ".txt", combined_preds, fmt="%.8f")
    with zipfile.ZipFile(file_name + ".zip", "w") as f:
        f.write(file_name + ".txt")

    print(f"Prediction time: {time()-start}")
