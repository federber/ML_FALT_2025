"""test_losses

"""

import pytest
import torch
from src.utils.losses import SimCLR_Loss

def test_simclr_loss():
    criterion = SimCLR_Loss(temperature=0.5)

    # идентичные эмбеддинги → низкий loss
    z_i = torch.randn(4, 128)
    z_j = z_i.clone()
    loss = criterion(z_i, z_j)
    assert loss.item() < 1.0

    # случайные эмбеддинги → высокий loss
    z_j = torch.randn(4, 128)
    loss = criterion(z_i, z_j)
    assert loss.item() > 4.0

def test_loss_batch_size_1():
    criterion = SimCLR_Loss(temperature=0.5)
    z_i = torch.randn(1, 128)
    z_j = torch.randn(1, 128)
    with pytest.raises(Exception):
        _ = criterion(z_i, z_j)