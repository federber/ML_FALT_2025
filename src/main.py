from src.config import Config
from src.data import prepare_data
from src.model import EncoderDecoder
from src.utils import LabelSmoothingLoss, NoamOpt
from src.train import fit

import torch

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    word_field, train_iter, test_iter, pretrained_embeddings = prepare_data(Config, device)

    model = EncoderDecoder(
        source_vocab_size=len(word_field.vocab),
        target_vocab_size=len(word_field.vocab),
        d_model=Config.d_model,
        heads_count=Config.heads_count,
        pretrained_embeddings=pretrained_embeddings,
        use_shared_emb=False
    ).to(device)

    pad_idx = word_field.vocab.stoi["<pad>"]

    criterion = LabelSmoothingLoss(
        smoothing=0.05,
        vocab_size=len(word_field.vocab),
        ignore_index=pad_idx
    ).to(device)

    optimizer = NoamOpt(model.d_model)

    fit(
        model, criterion, optimizer,
        train_iter,
        epochs_count=Config.epochs_count,
        val_iter=test_iter,
        save_path1=Config.model_path,
        patience=Config.patience,
        train_metrics_file=Config.train_metrics_file,
        val_metrics_file=Config.val_metrics_file
    )