
import torch
from model import Model

def test_model_output_shape():
    model = Model()
    dummy_input = torch.rand((1, 4, 84, 84))  # 1 образец, 4 канала, 84x84 изображение
    output = model(dummy_input)
    assert output.shape == (1, 2), "Model output must have shape (1, 2) for 2 actions"

def test_model_weights_init():
    model = Model()
    conv_layers = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
    for layer in conv_layers:
        assert layer.weight is not None, "Conv2d layer weights must be initialized"
