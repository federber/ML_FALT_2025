import torch
from torch import nn
from torch.utils.data import DataLoader, SubsetRandomSampler
import pandas as pd
import numpy as np
from torchvision import models
import wandb
import csv
import matplotlib.pyplot as plt

from config import (
    DEVICE,
    BATCH_SIZE,
    LR,
    WEIGHT_DECAY,
    NUM_EPOCHS,
    MARGIN,
    TRIPLET_TRAIN_CSV,
    TRIPLET_VALID_CSV,
    MODEL_CHECKPOINT,
    TRAIN_LOG_CSV,
)
from datasets import FaceDatasetTriplet


def train_model():
    # Устройство (CPU или GPU)
    device = torch.device(DEVICE)
    print(f"Using device: {device}")

    # Загружаем датасеты
    triplet_train_df = pd.read_csv(TRIPLET_TRAIN_CSV)
    triplet_valid_df = pd.read_csv(TRIPLET_VALID_CSV)

    train_triplet_dataset = FaceDatasetTriplet(triplet_train_df.sample(n=100000, random_state=42))
    valid_triplet_dataset = FaceDatasetTriplet(triplet_valid_df.sample(n=15000, random_state=42))

    train_loader = DataLoader(
        train_triplet_dataset,
        sampler=SubsetRandomSampler(range(len(train_triplet_dataset))),
        batch_size=BATCH_SIZE,
        num_workers=4,
        drop_last=True,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        valid_triplet_dataset,
        sampler=SubsetRandomSampler(range(len(valid_triplet_dataset))),
        batch_size=BATCH_SIZE,
        num_workers=4,
        drop_last=True,
        pin_memory=True,
    )

    # Инициализация модели (ResNet18 → Identity)
    encoder = models.resnet18(pretrained=True)
    encoder.fc = nn.Identity()
    encoder = encoder.to(device)

    criterion = nn.TripletMarginLoss(margin=MARGIN)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Инициализация WandB
    wandb.init(
        project="face-recognition-hw5",
        config={
            "architecture": "ResNet18",
            "loss": "TripletMarginLoss",
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "weight_decay": WEIGHT_DECAY,
            "margin": MARGIN,
            "epochs": NUM_EPOCHS,
        },
    )

    # Настройка CSV-логирования
    TRAIN_LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(TRAIN_LOG_CSV, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["epoch", "train_loss", "valid_loss", "learning_rate"])

    history = {"epoch": [], "train_loss": [], "valid_loss": [], "learning_rate": []}

    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    max_epochs_without_improvement = 1

    for epoch in range(1, NUM_EPOCHS + 1):
        # --------------------------
        #  Тренировка для одной эпохи
        # --------------------------
        encoder.train()
        running_train_loss = 0.0

        for idx, (anchor, positive, negative) in enumerate(train_loader):
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            optimizer.zero_grad()
            anchor_embed = encoder(anchor)
            positive_embed = encoder(positive)
            negative_embed = encoder(negative)

            loss_value = criterion(anchor_embed, positive_embed, negative_embed)
            loss_value.backward()
            optimizer.step()

            running_train_loss = (running_train_loss * idx + loss_value.item()) / (idx + 1)

        train_loss_epoch = running_train_loss

        # --------------------------
        #  Валидация для одной эпохи
        # --------------------------
        encoder.eval()
        running_valid_loss = 0.0
        total_valid_loss = 0.0

        with torch.no_grad():
            for idx, (anchor, positive, negative) in enumerate(valid_loader):
                anchor = anchor.to(device)
                positive = positive.to(device)
                negative = negative.to(device)

                anchor_embed = encoder(anchor)
                positive_embed = encoder(positive)
                negative_embed = encoder(negative)

                loss_value = criterion(anchor_embed, positive_embed, negative_embed)
                total_valid_loss += loss_value.item()
                running_valid_loss = (running_valid_loss * idx + loss_value.item()) / (idx + 1)

        valid_loss_epoch = total_valid_loss / len(valid_loader)

        current_lr = optimizer.param_groups[0]["lr"]

        # Логирование в WandB и историю
        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss_epoch,
                "valid_loss": valid_loss_epoch,
                "learning_rate": current_lr,
            }
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss_epoch)
        history["valid_loss"].append(valid_loss_epoch)
        history["learning_rate"].append(current_lr)

        with open(TRAIN_LOG_CSV, "a", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(
                [epoch, f"{train_loss_epoch:.6f}", f"{valid_loss_epoch:.6f}", f"{current_lr:.6e}"]
            )

        # Сохраняем чекпоинт при улучшении валидационной потери
        if valid_loss_epoch < best_validation_loss:
            MODEL_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"model_state_dict": encoder.state_dict(), "optimizer_state_dict": optimizer.state_dict()},
                MODEL_CHECKPOINT,
            )
            best_validation_loss = valid_loss_epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement > max_epochs_without_improvement:
                print("Early stopping triggered.")
                break

    wandb.finish()

    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["train_loss"], label="Train Loss")
    plt.plot(history["epoch"], history["valid_loss"], label="Valid Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.grid()
    plt.legend()
    plt.savefig("loss_curve.png")
    plt.close()

    plt.figure(figsize=(8, 3))
    plt.plot(history["epoch"], history["learning_rate"], label="Learning Rate")
    plt.xlabel("Epoch")
    plt.ylabel("LR")
    plt.title("Learning Rate Schedule")
    plt.grid()
    plt.legend()
    plt.savefig("lr_schedule.png")
    plt.close()


if __name__ == "__main__":
    train_model()