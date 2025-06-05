import torch
import pytest
from pathlib import Path

from src.model import get_encoder

def test_get_encoder_output_shape():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = get_encoder(pretrained=False, device=device)
    encoder.eval()

    #  dummy batch of images: batch_size=2, 3 channels, 224x224
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    with torch.no_grad():
        output = encoder(dummy_input)

    assert output.shape == (2, 512)

def test_encoder_non_trained_parameters():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = get_encoder(pretrained=False, device=device)

    num_params = sum(p.numel() for p in encoder.parameters())
    assert num_params > 0

def test_checkpoint_loading(tmp_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = get_encoder(pretrained=False, device=device)

    # dummy checkpoint
    checkpoint_path = tmp_path / "dummy_model.pt"
    torch.save({"model_state_dict": encoder.state_dict()}, checkpoint_path)

    loaded = torch.load(checkpoint_path, map_location=device)
    encoder_loaded = get_encoder(pretrained=False, device=device)
    encoder_loaded.load_state_dict(loaded["model_state_dict"])

    encoder.eval()
    encoder_loaded.eval()
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        out1 = encoder(dummy_input)
        out2 = encoder_loaded(dummy_input)
    assert torch.allclose(out1, out2)