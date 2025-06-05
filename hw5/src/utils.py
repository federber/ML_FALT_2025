import cv2
import numpy as np
from pathlib import Path
import pandas as pd


def get_celeba_paths(celeba_root: str):

    celeba_dir = Path(celeba_root)
    base_data_path = celeba_dir / "img_align_celeba"
    identity_file = celeba_dir / "identity_CelebA.txt"
    partition_file = celeba_dir / "list_eval_partition.txt"
    return base_data_path, identity_file, partition_file


def imread(path: str) -> np.ndarray:

    im = cv2.imread(str(path))
    if im is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def build_id_to_paths(df: pd.DataFrame) -> dict:

    id_paths = {int(i): [] for i in np.unique(df["id"].values)}
    for _, row in df.iterrows():
        id_paths[int(row["id"])].append(row["path"])
    return id_paths