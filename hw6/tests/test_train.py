
from train import TrainingSession

def test_training_session_init():
    trainer = TrainingSession()
    assert trainer.model is not None
    assert hasattr(trainer, 'optimizer')
    assert hasattr(trainer, 'loss_fn')
    assert trainer.memory == [], "Memory buffer should be empty at init"

def test_game_environment_is_ready():
    trainer = TrainingSession()
    observation = trainer.game.frame_step(0)[0]  # получаем кадр
    assert observation is not None
    assert observation.shape[-1] == 3, "Observation should have 3 channels (RGB)"
