import torch
import torch.nn as nn
from torchvision import models

def get_encoder(pretrained: bool = True, device: torch.device = torch.device("cpu")) -> nn.Module:
    encoder = models.resnet18(pretrained=pretrained)
    encoder.fc = nn.Identity()
    return encoder.to(device)