import os
import numpy as np
import pandas as pd
import pytest
import cv2

from utils import get_celeba_paths, imread, build_id_to_paths
from pathlib import Path

def test_get_celeba_paths(tmp_path):
    celeba_root = tmp_path / "celeba_test"
    img_dir = celeba_root / "img_align_celeba"
    identity_file = celeba_root / "identity_CelebA.txt"
    partition_file = celeba_root / "list_eval_partition.txt"
    output_dir = celeba_root

    img_dir.mkdir(parents=True)
    identity_file.write_text("dummy
")
    partition_file.write_text("dummy
")

    base_data_path, id_file, part_file, out_dir = get_celeba_paths(str(celeba_root))

    assert Path(base_data_path) == img_dir
    assert Path(id_file) == identity_file
    assert Path(part_file) == partition_file
    assert Path(out_dir) == output_dir

def test_imread_success(tmp_path):
    img_array = np.zeros((10, 10, 3), dtype=np.uint8)
    img_array[:] = [123, 222, 56]
    img_path = tmp_path / "test_image.jpg"
    cv2.imwrite(str(img_path), img_array)

    rgb = imread(str(img_path))

    assert rgb.shape == (10, 10, 3)
    assert np.array_equal(rgb[0, 0], np.array([56, 222, 123], dtype=np.uint8))

def test_imread_file_not_found(tmp_path):
    non_existent = tmp_path / "no_such_file.png"
    with pytest.raises(FileNotFoundError):
        imread(str(non_existent))

def test_build_id_to_paths():
    data = {
        "path": [
            "/path/to/img1.jpg",
            "/path/to/img2.jpg",
            "/another/path/img3.jpg",
            "/another/path/img4.jpg"
        ],
        "id": [1, 1, 2, 2]
    }
    df = pd.DataFrame(data)
    mapping = build_id_to_paths(df)

    assert set(mapping.keys()) == {1, 2}
    assert mapping[1] == ["/path/to/img1.jpg", "/path/to/img2.jpg"]
    assert mapping[2] == ["/another/path/img3.jpg", "/another/path/img4.jpg"]

def test_build_id_to_paths_single_entry():
    data = {"path": ["/only/one.jpg"], "id": [5]}
    df = pd.DataFrame(data)
    mapping = build_id_to_paths(df)
    assert mapping == {5: ["/only/one.jpg"]}
