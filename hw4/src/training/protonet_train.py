
import os
import torch
import wandb
from torch.optim import Adam
from torch.utils.data import DataLoader
from src.models import ProtoNet, Encoder
from src.utils.metrics import evaluate
from src.data import OmniglotDataset, ProtoTransform

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = os.path.join(os.path.dirname(__file__), "../../../models")

def train_protonet(n_way=5, k_shot=5, epochs=50, lr=1e-3):
    """Обучение ProtoNet на Omniglot"""
    encoder = Encoder().to(DEVICE)
    model = ProtoNet(encoder).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=lr)

    omniglot_train = OmniglotDataset(mode='train', transform=ProtoTransform())
    train_loader = DataLoader(omniglot_train, batch_size=20, shuffle=True)

    wandb.init(
        project="protonet-omniglot",
        config={
            "n_way": n_way,
            "k_shot": k_shot,
            "epochs": epochs,
            "lr": lr
        }
    )

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            support, query = batch
            support = support.to(DEVICE)
            query = query.to(DEVICE)

            optimizer.zero_grad()
            loss, acc = model(support, query, n_way, k_shot)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()


            wandb.log({
                "train_loss": loss.item(),
                "train_acc": acc
            })

        val_acc = evaluate(model, n_way, k_shot)
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs(SAVE_PATH, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(SAVE_PATH, "protonet_best.pt"))

        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f}")
        wandb.log({"val_acc": val_acc})

    wandb.finish()
    return model

