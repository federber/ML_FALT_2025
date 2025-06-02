
install:
	pip install -r requirements.txt

train-simclr:
	dvc repro train_simclr

train-protonet:
	dvc repro train_protonet

train-all: train-simclr train-protonet


report:
    python -m src.scripts.generate_report  

test:
	pytest tests/ -v

clean:
	dvc destroy
	rm -rf .dvc data/processed models/*.pt results/*.json

.PHONY: install test clean
