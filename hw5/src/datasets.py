import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2 as ToTensor

def imread(path: str) -> np.ndarray:
    im = cv2.imread(path)
    if im is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

class FaceDatasetTriplet(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.data = df.reset_index(drop=True)
        self.transforms = A.Compose([A.Normalize(), ToTensor()])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        anchor_path, positive_path, negative_path = self.data.iloc[idx]
        anchor_img = self.transforms(image=imread(anchor_path))['image']
        positive_img = self.transforms(image=imread(positive_path))['image']
        negative_img = self.transforms(image=imread(negative_path))['image']
        return anchor_img, positive_img, negative_img

class FaceDatasetVerification(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.data = df.reset_index(drop=True)
        self.transforms = A.Compose([A.Normalize(), ToTensor()])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        lhs_path, rhs_path, same = self.data.iloc[idx]
        lhs_img = self.transforms(image=imread(lhs_path))['image']
        rhs_img = self.transforms(image=imread(rhs_path))['image']
        return lhs_img, rhs_img, same