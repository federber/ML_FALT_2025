import math
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

class NoamOpt:
    def __init__(self, model_size, factor=2, warmup=4000, optimizer=None):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam([], lr=0)

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * (
            self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5))
        )


class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1, vocab_size=None, ignore_index=-100):
        super().__init__()
        self.smoothing = smoothing
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.vocab_size - 2))
            mask = (target != self.ignore_index).unsqueeze(1)
            true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - self.smoothing)
            true_dist = true_dist * mask
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))
