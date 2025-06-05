import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, SubsetRandomSampler
from model import get_encoder
from datasets import FaceDatasetVerification
from pathlib import Path
import json


def load_model(model_path: Path, device: torch.device):
    encoder = get_encoder().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    encoder.load_state_dict(checkpoint["model_state_dict"])
    encoder.eval()
    return encoder


def compute_distances_and_threshold(encoder, valid_csv: Path, device: torch.device):
    df = pd.read_csv(valid_csv)
    dataset = FaceDatasetVerification(df)
    loader = DataLoader(
        dataset,
        sampler=SubsetRandomSampler(range(len(dataset))),
        batch_size=1,
        num_workers=0,
        drop_last=False,
        pin_memory=True,
    )

    distances = np.zeros(len(dataset))
    targets = np.zeros(len(dataset))

    with torch.no_grad():
        for idx, (lhs, rhs, target) in enumerate(loader):
            lhs = lhs.to(device)
            rhs = rhs.to(device)
            lhs_embed = encoder(lhs)
            rhs_embed = encoder(rhs)
            distances[idx] = (lhs_embed - rhs_embed).pow(2).sum().sqrt().item()
            targets[idx] = target.item()

    num_steps = 20000
    tpr = np.zeros(num_steps)
    fpr = np.zeros(num_steps)
    precision = np.zeros(num_steps)
    recall = np.zeros(num_steps)

    thresholds = np.linspace(distances.min() + 0.01, distances.max() - 0.01, num_steps)
    for i, thresh in enumerate(thresholds):
        preds = (distances < thresh).astype(int)
        tp = ((preds == 1) & (targets == 1)).sum()
        fp = ((preds == 1) & (targets == 0)).sum()
        tn = ((preds == 0) & (targets == 0)).sum()
        fn = ((preds == 0) & (targets == 1)).sum()

        tpr[i] = tp / (tp + fn + 1e-8)
        fpr[i] = fp / (fp + tn + 1e-8)
        precision[i] = tp / (tp + fp + 1e-8)
        recall[i] = tp / (tp + fn + 1e-8)

    f1_score = 2 * precision * recall / (precision + recall + 1e-8)
    beta = 0.7
    f_beta = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-8)

    best_thresh_f1 = float(thresholds[np.argmax(f1_score)])
    best_thresh_fb = float(thresholds[np.argmax(f_beta)])

    # Сохранить кривые для дальнейшей визуализации
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.grid()
    plt.savefig("roc_curve.png")
    plt.close()

    plt.figure()
    plt.plot(thresholds, f1_score)
    plt.xlabel("Threshold")
    plt.ylabel("F1-Score")
    plt.title("F1-Score Curve")
    plt.grid()
    plt.savefig("f1_curve.png")
    plt.close()

    return {
        "best_threshold_f1": best_thresh_f1,
        "best_threshold_fb": best_thresh_fb,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--valid_csv", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    encoder = load_model(Path(args.model), device)
    results = compute_distances_and_threshold(encoder, Path(args.valid_csv), device)

    # Сохраняем пороги в JSON
    with open("thresholds.json", "w") as f:
        json.dump(results, f)

    print(f"Selected threshold (F1): {results['best_threshold_f1']}")
    print(f"Selected threshold (Fβ): {results['best_threshold_fb']}")