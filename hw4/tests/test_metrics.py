"""test_metrics

"""

import torch
from src.utils.evaluation import evaluate

def test_accuracy():
    model = torch.nn.Linear(10, 2)
    X = torch.randn(10, 10)
    y = torch.randint(0, 2, (10,))

    loader = [(X, y)]
    acc = evaluate(model, loader)
    assert 0.0 <= acc <= 1.0

class DummyModel(torch.nn.Module):
    def forward(self, x):
        return torch.randn(x.shape[0], 2)

def test_evaluate_dummy():
    model = DummyModel()
    X = torch.randn(5, 10)
    y = torch.randint(0, 2, (5,))
    loader = [(X, y)]
    acc = evaluate(model, loader)
    assert 0.0 <= acc <= 1.0