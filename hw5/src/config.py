
import os
from pathlib import Path
import torch

# Пути
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "intermediate"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Датасеты
CELEBA_DIR = RAW_DIR / "celeba"
IDENTITY_FILE = CELEBA_DIR / "identity_CelebA.txt"
PARTITION_FILE = CELEBA_DIR / "list_eval_partition.txt"

# Параметры обучения
BATCH_SIZE = 16
LR = 1e-4
WEIGHT_DECAY = 1e-2
NUM_EPOCHS = 10
MARGIN = 1.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Файлы промежуточных данных
CELEBA_ID_CSV = INTERIM_DIR / "celeba_id.csv"
PARTITION_CSV = INTERIM_DIR / "list_eval_partition.csv"
TRIPLET_TRAIN_CSV = INTERIM_DIR / "triplet_train.csv"
TRIPLET_VALID_CSV = INTERIM_DIR / "triplet_valid.csv"
VERIF_VALID_CSV = INTERIM_DIR / "verification_valid.csv"
VERIF_TEST_CSV = INTERIM_DIR / "verification_test.csv"

# Модель
MODEL_CHECKPOINT = MODELS_DIR / "verification_model.pt"
TRAIN_LOG_CSV = LOGS_DIR / "training_log.csv"
THRESHOLD_JSON = INTERIM_DIR / "threshold.json"
