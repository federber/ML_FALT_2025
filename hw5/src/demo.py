
import json
import numpy as np
import pandas as pd
import torch
import cv2
from torchvision import models
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2 as ToTensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data_id = pd.read_csv("data/intermediate/train_data_id.csv")
test_data_id = pd.read_csv("data/intermediate/test_data_id.csv")

assert set(test_data_id.id).isdisjoint(set(train_data_id.id))

uniques, counts = np.unique(test_data_id.id, return_counts=True)
valid_test_ids = uniques[counts > 15]
test_data_id = test_data_id[test_data_id.id.isin(valid_test_ids)]

chosen_id = np.random.choice(valid_test_ids, 1)[0]
reference_row = test_data_id[test_data_id.id == chosen_id].iloc[0]
same_id_rows = test_data_id[test_data_id.id == chosen_id].iloc[1:6]
other_rows = test_data_id[test_data_id.id != chosen_id].sample(10)
test_paths = pd.concat([same_id_rows, other_rows], ignore_index=True)

encoder = models.resnet18(pretrained=True).to(device)
encoder.fc = torch.nn.Identity()
checkpoint = torch.load("verification_model.pt", map_location=device)
encoder.load_state_dict(checkpoint['model_state_dict'])
encoder.eval()

with open("data/intermediate/threshold.json", "r") as f:
    threshold = json.load(f)["threshold"]

transforms = A.Compose([A.Normalize(), ToTensor()])

ref_img = cv2.cvtColor(cv2.imread(reference_row.path), cv2.COLOR_BGR2RGB)
ref_tensor = transforms(image=ref_img)["image"].unsqueeze(0).to(device)
ref_embed = encoder(ref_tensor)

tp = tn = fp = fn = 0
for _, row in test_paths.iterrows():
    img = cv2.cvtColor(cv2.imread(row.path), cv2.COLOR_BGR2RGB)
    tensor = transforms(image=img)["image"].unsqueeze(0).to(device)
    embed = encoder(tensor)
    distance = torch.norm(ref_embed - embed).item()
    pred = int(distance < threshold)
    target = int(row.id == reference_row.id)
    if pred == 1 and target == 1:
        tp += 1
    elif pred == 1 and target == 0:
        fp += 1
    elif pred == 0 and target == 0:
        tn += 1
    elif pred == 0 and target == 1:
        fn += 1

print(f"Selected reference ID: {chosen_id}")
print(f"Out of 10 different-ID images, correctly flagged as different: {tn}")
print(f"Out of 5 same-ID images, correctly flagged as same:      {tp}")
