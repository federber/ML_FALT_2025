"""test_data

"""

import pytest
import torch
from src.data.datasets import get_cifar10
from src.data.transforms import simclr_transform

def test_dataset_augmentations():
    dataset = get_cifar10(train=True)
    img1, img2 = dataset[0]

    assert img1.shape == (3, 32, 32)
    assert img2.shape == (3, 32, 32)

    assert torch.all(img1 >= -1.0) and torch.all(img1 <= 1.0)

def test_dataset_split():
    train_dataset = get_cifar10(train=True)
    test_dataset = get_cifar10(train=False)
    assert len(train_dataset) == 50000
    assert len(test_dataset) == 10000

def test_dataset_len_and_type():
    dataset = get_cifar10(train=True)
    assert isinstance(dataset[0][0], torch.Tensor)
    assert isinstance(dataset[0][1], torch.Tensor)
    assert len(dataset) > 1000  # sanity check