"""test_pipeline
"""

def test_simclr_training():
    from src.training.simclr_train import train_simclr
    model = train_simclr(epochs=1, debug=True)
    assert model is not None

def test_protonet_training():
    from src.training.protonet_train import train_protonet
    model, logs = train_protonet(epochs=1, debug=True)
    assert hasattr(model, "forward")
    assert logs["train_loss"][-1] < 5.0