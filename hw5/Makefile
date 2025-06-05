
.PHONY: all prepare build_triplets build_verification train evaluate test demo clean

all: train

# 1: Prepare data (identity CSV + partition CSV)
prepare:
	python src/data_preparation.py

# 2: Build triplet datasets (train + validation)
build_triplets: prepare
	python src/triplet_builder.py

# 3: Build verification datasets (valid + test)
build_verification: build_triplets
	python src/verification_builder.py

# 4: Train model
train: build_verification
	python src/train.py

# 5: Evaluate on validation set and produce threshold
evaluate: train
	python src/evaluate.py --model verification_model.pt --valid_csv data/intermediate/verification_valid.csv

# 6: Test model on test set
test: evaluate
	python src/test_model.py

# 7: Demo on sample images
demo: evaluate
	python src/demo.py

clean:
	rm -f verification_model.pt
	rm -f training_log.csv
	rm -f data/intermediate/*.csv
	rm -f data/intermediate/threshold.json
	rm -f *.png
