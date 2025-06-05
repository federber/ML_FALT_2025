import numpy as np
import pandas as pd
import cv2
import pytest
from pathlib import Path
import torch

from datasets import FaceDatasetTriplet, FaceDatasetVerification, imread

def test_face_dataset_triplet(tmp_path):
    # dummy images
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(4):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        img[:] = [i * 10, i * 20, i * 30]
        cv2.imwrite(str(img_dir / f"img{i}.jpg"), img)

    # df for triplets
    data = {
        "anchor": [str(img_dir / "img0.jpg"), str(img_dir / "img1.jpg")],
        "positive": [str(img_dir / "img1.jpg"), str(img_dir / "img2.jpg")],
        "negative": [str(img_dir / "img2.jpg"), str(img_dir / "img3.jpg")],
    }
    df = pd.DataFrame(data)
    dataset = FaceDatasetTriplet(df)

    assert len(dataset) == 2

    anchor_tensor, pos_tensor, neg_tensor = dataset[0]
    assert isinstance(anchor_tensor, torch.Tensor)
    assert anchor_tensor.shape[1:] == (10, 10)  # 3x10x10

def test_face_dataset_verification(tmp_path):
    # dummy images
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(4):
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        img[:] = [i * 15, i * 25, i * 35]
        cv2.imwrite(str(img_dir / f"img{i}.jpg"), img)

    # df for verification pairs
    data = {
        "lhs_path": [str(img_dir / "img0.jpg"), str(img_dir / "img2.jpg")],
        "rhs_path": [str(img_dir / "img1.jpg"), str(img_dir / "img3.jpg")],
        "same": [1, 0],
    }
    df = pd.DataFrame(data)
    dataset = FaceDatasetVerification(df)

    assert len(dataset) == 2

    lhs_tensor, rhs_tensor, label = dataset[1]
    assert isinstance(lhs_tensor, torch.Tensor)
    assert lhs_tensor.shape[1:] == (8, 8)  # 3x8x8
    assert label == 0
