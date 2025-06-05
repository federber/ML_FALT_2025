
import torch
import numpy as np
import pytest
from eval import EvaluationSession
from model import Model, GameProcessor

@pytest.fixture
def dummy_model_path(tmp_path):
    model = Model()
    path = tmp_path / "dummy_model.pt"
    torch.save(model, path)
    return path

def test_eval_session_init(dummy_model_path):
    session = EvaluationSession(model_path=dummy_model_path)
    assert session.model is not None
    assert hasattr(session, 'game'), "Game should be initialized"

def test_prepare_initial_state_shape(dummy_model_path):
    session = EvaluationSession(model_path=dummy_model_path)
    frame = np.zeros((288, 512, 3), dtype=np.uint8)
    state = session._prepare_initial_state(frame)
    assert state.shape == (1, 4, 84, 84), "State must be of shape (1, 4, 84, 84)"
