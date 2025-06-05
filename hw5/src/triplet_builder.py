import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

NUM_ROUNDS = 9
NUM_TRIPLETS_PER_ROUND = 70000

CELEBA_ID_CSV = Path("data/intermediate/celeba_id.csv")
PARTITION_CSV = Path("data/intermediate/list_eval_partition.csv")
TRIPLET_TRAIN_CSV = Path("data/intermediate/triplet_train.csv")
TRIPLET_VALID_CSV = Path("data/intermediate/triplet_valid.csv")


def build_id_to_paths(df: pd.DataFrame) -> dict:
    id_paths = {i: [] for i in np.unique(df["id"].values)}
    for _, row in df.iterrows():
        id_paths[row["id"]].append(row["path"])
    return id_paths


def main():
    celeba_df = pd.read_csv(CELEBA_ID_CSV)
    partition_df = pd.read_csv(PARTITION_CSV)

    celeba_df["filename"] = celeba_df["path"].apply(lambda x: Path(x).name)
    merged_df = celeba_df.merge(partition_df, on="filename")

    train_df = merged_df.loc[merged_df["partition"] == 0].reset_index(drop=True)
    valid_df = merged_df.loc[merged_df["partition"] == 1].reset_index(drop=True)

    id_to_paths_train = build_id_to_paths(train_df[["path", "id"]])
    id_to_paths_valid = build_id_to_paths(valid_df[["path", "id"]])

    triplet_train_list = []
    ids_train = list(id_to_paths_train.keys())
    for _ in range(NUM_ROUNDS):
        for _ in tqdm(range(NUM_TRIPLETS_PER_ROUND), desc="Building train triplets"):
            while True:
                anchor_id, negative_id = np.random.choice(ids_train, 2, replace=False)
                if len(id_to_paths_train[anchor_id]) > 1:
                    break
            anchor_list = id_to_paths_train[anchor_id]
            anchor_idx = np.random.randint(0, len(anchor_list))
            anchor_path = anchor_list[anchor_idx]

            positive_candidates = anchor_list[:anchor_idx] + anchor_list[anchor_idx + 1 :]
            positive_path = np.random.choice(positive_candidates)
            negative_path = np.random.choice(id_to_paths_train[negative_id])

            triplet_train_list.append(
                {"anchor": anchor_path, "positive": positive_path, "negative": negative_path}
            )

    triplet_train_df = pd.DataFrame(triplet_train_list)
    triplet_train_df.to_csv(TRIPLET_TRAIN_CSV, index=False)

    triplet_valid_list = []
    ids_valid = list(id_to_paths_valid.keys())
    for _ in tqdm(range(NUM_TRIPLETS_PER_ROUND), desc="Building valid triplets"):
        while True:
            anchor_id, negative_id = np.random.choice(ids_valid, 2, replace=False)
            if len(id_to_paths_valid[anchor_id]) > 1:
                break
        anchor_list = id_to_paths_valid[anchor_id]
        anchor_idx = np.random.randint(0, len(anchor_list))
        anchor_path = anchor_list[anchor_idx]

        positive_candidates = anchor_list[:anchor_idx] + anchor_list[anchor_idx + 1 :]
        positive_path = np.random.choice(positive_candidates)
        negative_path = np.random.choice(id_to_paths_valid[negative_id])

        triplet_valid_list.append(
            {"anchor": anchor_path, "positive": positive_path, "negative": negative_path}
        )

    triplet_valid_df = pd.DataFrame(triplet_valid_list)
    triplet_valid_df.to_csv(TRIPLET_VALID_CSV, index=False)


if __name__ == "__main__":
    main()