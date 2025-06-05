import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import models
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2 as ToTensor

class FaceDatasetVerification(torch.utils.data.Dataset):
    def __init__(self, df):
        self.data = df.reset_index(drop=True)
        self.transforms = A.Compose([
            A.Normalize(),
            ToTensor()
        ])
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        lhs_path, rhs_path, same = self.data.iloc[idx]
        lhs_img = self.transforms(image=__import__('cv2').cv2.imread(lhs_path)[:,:,::-1])['image']
        rhs_img = self.transforms(image=__import__('cv2').cv2.imread(rhs_path)[:,:,::-1])['image']
        return lhs_img, rhs_img, same

def test_model_performance(
    model_path: str,
    threshold_json: str,
    test_csv: str,
    device: torch.device
):
    # Загрузить модель
    encoder = models.resnet18(pretrained=True).to(device)
    encoder.fc = torch.nn.Identity()
    checkpoint = torch.load(model_path, map_location=device)
    encoder.load_state_dict(checkpoint["model_state_dict"])
    encoder.eval()

    import json
    with open(threshold_json, "r") as f:
        threshold = json.load(f).get("threshold", json.load(f)["best_threshold_f1"])

    df_test = pd.read_csv(test_csv)
    test_dataset = FaceDatasetVerification(df_test)
    test_loader = DataLoader(
        test_dataset,
        sampler=SubsetRandomSampler(range(len(test_dataset))),
        batch_size=1,
        num_workers=0,
        drop_last=False,
        pin_memory=True
    )

    tp = fp = tn = fn = 0
    with torch.no_grad():
        for lhs, rhs, target in test_loader:
            lhs = lhs.to(device)
            rhs = rhs.to(device)
            lhs_embed = encoder(lhs)
            rhs_embed = encoder(rhs)
            distance = (lhs_embed - rhs_embed).pow(2).sum().sqrt().item()
            pred = int(distance < threshold)
            t = int(target.item())
            if pred == 1 and t == 1:
                tp += 1
            elif pred == 1 and t == 0:
                fp += 1
            elif pred == 0 and t

precision_test = tp / (tp + fp + 1e-8)
recall_test = tp / (tp + fn + 1e-8)
f1_test = 2 * precision_test * recall_test / (precision_test + recall_test + 1e-8)
print(f"Test Recall:    {recall_test:.4f}")
print(f"Test Precision: {precision_test:.4f}")
print(f"Test F1-Score:  {f1_test:.4f}")
