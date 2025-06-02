

import os
import torch
import wandb
import matplotlib.pyplot as plt
import pandas as pd
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from src.data import CIFAR10Dataset, TwoAugDataset, simclr_transform
from src.models import SimCLRModel, SimCLR_Loss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = os.path.join(os.path.dirname(__file__), "../../../models")

def train_simclr(epochs=100, batch_size=256, lr=3e-4):
    """Обучение модели SimCLR"""
    model = SimCLRModel().to(DEVICE)
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = SimCLR_Loss(temperature=0.5).to(DEVICE)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=3, verbose=True)

    cifar_train = CIFAR10Dataset.create(train=True, transform=None)
    train_loader = DataLoader(
        TwoAugDataset(cifar_train, simclr_transform),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    wandb.init(
        project="simclr-cifar10",
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "architecture": "SimCLR"
        }
    )

    best_loss = float('inf')
    loss_history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for x1, x2 in train_loader:
            x1, x2 = x1.to(DEVICE), x2.to(DEVICE)

            optimizer.zero_grad()
            z_i = model(x1)
            z_j = model(x2)
            loss = criterion(z_i, z_j)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)
        loss_history.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs(SAVE_PATH, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(SAVE_PATH, "simclr_best.pt"))

        wandb.log({"epoch": epoch+1, "loss": avg_loss})
        print(f"Epoch {epoch+1}/{epochs}: Loss {avg_loss:.4f}")

    results_path = os.path.join(os.path.dirname(__file__), "../../../results")
    os.makedirs(results_path, exist_ok=True)

    plt.figure(figsize=(10,5))
    plt.plot(loss_history)
    plt.title("SimCLR Training Loss")
    plt.savefig(os.path.join(results_path, "simclr_loss.png"))

    pd.DataFrame({"epoch": range(1, len(loss_history)+1), "loss": loss_history}).to_csv(
        os.path.join(results_path, "simclr_log.csv"), index=False
    )

    wandb.finish()
    return model

