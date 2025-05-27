import os
import pandas as pd
import pytest
from train import save_metrics
from pathlib import Path

# 📌 Пути до метрик
TEST_METRICS_CSV = Path("models/test_metrics.csv")
BASELINE_CSV = Path("models/baseline_metrics.csv")

def test_test_metrics_exist():
    assert TEST_METRICS_CSV.exists(), "Файл test_metrics.csv не найден — нужно сначала запустить evaluate_on_test()"

def test_test_metrics_content():
    df = pd.read_csv(TEST_METRICS_CSV)
    assert df.shape[0] >= 1, "Файл test_metrics.csv пуст"
    for col in ["loss", "ppx", "rouge1", "rouge2", "rougeL"]:
        assert col in df.columns, f"Метрика '{col}' отсутствует в test_metrics.csv"

    metrics = df.iloc[0]
    assert metrics["loss"] > 0 and metrics["loss"] < 2.5
    assert metrics["ppx"] > 1 and metrics["ppx"] < 15
    assert 0 <= metrics["rouge1"] <= 1
    assert 0 <= metrics["rouge2"] <= 1
    assert 0 <= metrics["rougeL"] <= 1
    assert metrics["rouge1"] > 0.2, "ROUGE-1 слишком низкий"
    assert metrics["rougeL"] > 0.2, "ROUGE-L слишком низкий"

def test_compare_with_baseline():
    if not BASELINE_CSV.exists():
        pytest.skip("Бейзлайн не найден — сравнение пропущено")

    baseline = pd.read_csv(BASELINE_CSV).iloc[0]
    test = pd.read_csv(TEST_METRICS_CSV).iloc[0]

    assert test["rouge1"] >= baseline["rouge1"], f"ROUGE-1 ниже бейзлайна: {test['rouge1']:.3f} < {baseline['rouge1']:.3f}"
    assert test["rougeL"] >= baseline["rougeL"], f"ROUGE-L ниже бейзлайна: {test['rougeL']:.3f} < {baseline['rougeL']:.3f}"
    assert test["loss"] <= baseline["loss"] + 0.01, f"Loss выше бейзлайна: {test['loss']:.3f} > {baseline['loss']:.3f}"