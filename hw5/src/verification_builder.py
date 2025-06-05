import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

from config import CELEBA_ID_CSV, PARTITION_CSV, VERIF_VALID_CSV, VERIF_TEST_CSV

def build_id_to_paths(df):
    id_paths = {i: [] for i in np.unique(df["id"].values)}
    for _, row in df.iterrows():
        id_paths[row["id"]].append(row["path"])
    return id_paths

def main():
    celeba_df = pd.read_csv(CELEBA_ID_CSV)
    partition_df = pd.read_csv(PARTITION_CSV)
    celeba_df["filename"] = celeba_df["path"].apply(lambda x: Path(x).name)
    merged = celeba_df.merge(partition_df, on="filename")

    valid_df = merged.loc[merged["partition"] == 1].reset_index(drop=True)
    test_df = merged.loc[merged["partition"] == 2].reset_index(drop=True)

    id_to_paths_valid = build_id_to_paths(valid_df[["path", "id"]])
    id_to_paths_test = build_id_to_paths(test_df[["path", "id"]])

    verification_valid = []
    ids_valid = list(id_to_paths_valid.keys())

    for _ in tqdm(range(5000), desc="Building valid positive pairs"):
        while True:
            id_ = np.random.choice(ids_valid)
            if len(id_to_paths_valid[id_]) > 1:
                break
        lhs, rhs = np.random.choice(id_to_paths_valid[id_], 2, replace=False)
        verification_valid.append([lhs, rhs, 1])

    for _ in tqdm(range(5000), desc="Building valid negative pairs"):
        id_lhs, id_rhs = np.random.choice(ids_valid, 2, replace=False)
        lhs = np.random.choice(id_to_paths_valid[id_lhs])
        rhs = np.random.choice(id_to_paths_valid[id_rhs])
        verification_valid.append([lhs, rhs, 0])

    df_valid = pd.DataFrame(verification_valid, columns=["lhs_path", "rhs_path", "same"])
    VERIF_VALID_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_valid.to_csv(VERIF_VALID_CSV, index=False)

    verification_test = []
    ids_test = list(id_to_paths_test.keys())

    for _ in tqdm(range(5000), desc="Building test positive pairs"):
        while True:
            id_ = np.random.choice(ids_test)
            if len(id_to_paths_test[id_]) > 1:
                break
        lhs, rhs = np.random.choice(id_to_paths_test[id_], 2, replace=False)
        verification_test.append([lhs, rhs, 1])

    for _ in tqdm(range(5000), desc="Building test negative pairs"):
        id_lhs, id_rhs = np.random.choice(ids_test, 2, replace=False)
        lhs = np.random.choice(id_to_paths_test[id_lhs])
        rhs = np.random.choice(id_to_paths_test[id_rhs])
        verification_test.append([lhs, rhs, 0])

    df_test = pd.DataFrame(verification_test, columns=["lhs_path", "rhs_path", "same"])
    VERIF_TEST_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_test.to_csv(VERIF_TEST_CSV, index=False)

if __name__ == "__main__":
    main()