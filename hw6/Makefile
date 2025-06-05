train:
	python3 main.py --mode train

eval:
	python3 main.py --mode eval

test:
	pytest tests/ --disable-warnings

test-model:
	pytest tests/test_model.py --disable-warnings

test-train:
	pytest tests/test_train.py --disable-warnings

test-eval:
	pytest tests/test_eval.py --disable-warnings


dvc-init:
	dvc init

dvc-run:
	dvc repro

dvc-status:
	dvc status

clean:
	rm -rf __pycache__ .pytest_cache *.pyc *.pth *.pyo *.pyd *.log logs/ wandb/ .mypy_cache .env

