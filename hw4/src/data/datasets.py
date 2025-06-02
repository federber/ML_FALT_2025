
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets

class OmniglotDataset:
    @staticmethod
    def read_alphabets(alphabet_directory_path, alphabet_name):
        datax, datay, metay = [], [], []
        for character in os.listdir(alphabet_directory_path):
            char_path = os.path.join(alphabet_directory_path, character)
            if not os.path.isdir(char_path):
                continue
            for img_file in os.listdir(char_path):
                if not img_file.endswith(".png"):
                    continue
                img_path = os.path.join(char_path, img_file)
                img = Image.open(img_path).convert('RGB')
                for angle, suffix in zip([0, 90, 180, 270], ["rot0", "rot90", "rot180", "rot270"]):
                    rotated_img = img.rotate(angle, expand=True)
                    datax.append(np.array(rotated_img))
                    datay.append(f"{alphabet_name}/{character}")
                    metay.append(suffix)
        return datax, datay, metay

    @staticmethod
    def read_images(base_directory):
        datax, datay, metay = [], [], []
        for directory in os.listdir(base_directory):
            alphabet_path = os.path.join(base_directory, directory)
            if not os.path.isdir(alphabet_path):
                continue
            dx, dy, dmeta = OmniglotDataset.read_alphabets(alphabet_path, directory)
            datax.extend(dx)
            datay.extend(dy)
            metay.extend(dmeta)
        return datax, datay, metay

class CIFAR10Dataset:
    @staticmethod
    def create(train=True, transform=None):
        return datasets.CIFAR10(
            root='./data',
            train=train,
            download=True,
            transform=transform
        )

class TwoAugDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        x1 = self.transform(img)
        x2 = self.transform(img)
        return x1, x2