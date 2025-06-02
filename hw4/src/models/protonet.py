

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.block(x)

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 64),
            ConvBlock(64, 64),
            ConvBlock(64, 64)
        )

    def forward(self, x):
        return self.encoder(x).view(x.size(0), -1)

class ProtoNet(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder.to(self.device)

    @staticmethod
    def euclidean_dist(x, y):
        return torch.cdist(x, y, p=2).pow(2)

    def set_forward_loss(self, sample):
        sample_images = sample['images'].to(self.device)
        n_way = sample['n_way']
        n_support = sample['n_support']
        n_query = sample['n_query']

        # Подготовка данных
        x_dim = sample_images.size(2)
        sample_images = sample_images.view(n_way, n_support + n_query, x_dim, 28, 28)
        support = sample_images[:, :n_support].contiguous().view(-1, x_dim, 28, 28)
        query = sample_images[:, n_support:].contiguous().view(-1, x_dim, 28, 28)

        z_support = self.encoder(support)
        z_query = self.encoder(query)

        z_proto = z_support.view(n_way, n_support, -1).mean(1)

        dists = self.euclidean_dist(z_query, z_proto)
        log_p_y = F.log_softmax(-dists, dim=1)

        y_query = torch.repeat_interleave(torch.arange(n_way), n_query).to(self.device)

        loss = F.nll_loss(log_p_y, y_query)
        acc = (log_p_y.argmax(dim=1) == y_query).float().mean()

        return loss, {
            'loss': loss.item(),
            'acc': acc.item(),
            'y_hat': log_p_y.argmax(dim=1)
        }

def load_protonet_conv(**kwargs):
    class Encoder(nn.Module):
        def __init__(self, input_dim=3, hid_dim=64, z_dim=64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(input_dim, hid_dim, 3, padding=1),
                nn.BatchNorm2d(hid_dim),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(hid_dim, hid_dim, 3, padding=1),
                nn.BatchNorm2d(hid_dim),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(hid_dim, hid_dim, 3, padding=1),
                nn.BatchNorm2d(hid_dim),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(hid_dim, z_dim, 3, padding=1),
                nn.BatchNorm2d(z_dim),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1)
            )

        def forward(self, x):
            return self.net(x).flatten(1)

    return ProtoNet(Encoder(**kwargs))