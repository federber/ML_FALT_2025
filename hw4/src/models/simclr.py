
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ProjectionHead(nn.Module):
    def __init__(self, in_dim=512, hid_dim=512, out_dim=128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim)
        )

    def forward(self, x):
        return F.normalize(self.proj(x), dim=1)

class SimCLRModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = models.resnet18(weights=None)
        self.encoder.fc = nn.Identity()
        self.projector = ProjectionHead(in_dim=512)

    def forward(self, x):
        features = self.encoder(x)
        return self.projector(features)

class SimCLR_Loss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.cosine_sim = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def forward(self, z_i, z_j):
        batch_size = z_i.size(0)

        z = torch.cat([z_i, z_j], dim=0)

        sim_matrix = self.cosine_sim(
            z.unsqueeze(1),
            z.unsqueeze(0)
        ) / self.temperature

        mask = torch.eye(2*batch_size, dtype=torch.bool, device=z.device)
        sim_matrix.masked_fill_(mask, -float('inf'))

        positives = torch.cat([
            sim_matrix.diag(batch_size),
            sim_matrix.diag(-batch_size)
        ], dim=0).view(2*batch_size, 1)

        negatives = sim_matrix[mask ^ 1].view(2*batch_size, -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(2*batch_size, dtype=torch.long, device=z.device)

        return self.criterion(logits, labels) / (2*batch_size)