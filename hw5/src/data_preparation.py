import csv
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import config

def read_identity_file():
    identity_file = config.IDENTITY_FILE
    images_dir = config.CELEBA_DIR / "img_align_celeba"
    output_path = config.CELEBA_ID_CSV

    output_path.parent.mkdir(parents=True, exist_ok=True)
    celeba_id_dataframe = pd.DataFrame(columns=["path", "id"])

    with open(identity_file, "r") as f:
        for line in tqdm(f, desc="Reading identity file"):
            line = line.strip()
            if not line:
                continue
            filename, id_str = line.split()
            img_path = images_dir / filename
            celeba_id_dataframe = pd.concat(
                [
                    celeba_id_dataframe,
                    pd.DataFrame({"path": [str(img_path)], "id": [int(id_str)]}),
                ],
                ignore_index=True,
            )

    celeba_id_dataframe.to_csv(output_path, index=False)
    print(f"Saved celeba_id.csv to: {output_path}")


def convert_partition_to_csv():
    input_file = config.PARTITION_FILE
    output_file = config.PARTITION_CSV

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(input_file, "r") as infile, open(output_file, "w", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["filename", "partition"])
        for line in tqdm(infile, desc="Converting partition file"):
            line = line.strip()
            if not line:
                continue
            filename, partition_idx = line.split()
            writer.writerow([filename, int(partition_idx)])

    print(f"Saved list_eval_partition.csv to: {output_file}")


def main():
    read_identity_file()
    convert_partition_to_csv()


if __name__ == "__main__":
    main()