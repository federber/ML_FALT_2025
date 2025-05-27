from pathlib import Path

class Config:
    save_path = "/content/drive/MyDrive/Colab Notebooks/коалиция_ниггавуманов/"
    model_path = save_path + "NLP_goyda_checkp1.pt"
    vocab_path = save_path + "vocab_nlp_pupupu.pt"
    word_field_path = save_path + "word_field_nlp_pupupu.pt"
    dataset_path = save_path + "dataset_examples_пупу.pkl"
    embedding_path = save_path + "cc.ru.300.vec"

    train_metrics_file = Path(save_path + "train_metrics.csv")
    val_metrics_file = Path(save_path + "val_metrics.csv")

    d_model = 512
    d_ff = 1024
    heads_count = 8
    blocks_count = 4
    dropout = 0.1
    epochs_count = 30
    batch_size = 32
    patience = 7
    teacher_forcing_start = 0.7
    teacher_forcing_min = 0.1
    teacher_forcing_decay = 0.03