import math
import random
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import pandas as pd
from rouge_score import rouge_scorer
from pathlib import Path
from src.data import convert_batch


def decode_tokens(tensor, vocab, skip_tokens=("<pad>", "<s>", "</s>")):
    sentences = []
    skip_ids = [vocab.stoi[token] for token in skip_tokens if token in vocab.stoi]
    for seq in tensor:
        tokens = [vocab.itos[token.item()] for token in seq if token.item() not in skip_ids]
        sentences.append(" ".join(tokens))
    return sentences


def do_epoch(model, criterion, data_iter, optimizer=None, name=None, teacher_forcing_ratio=0.5):
    epoch_loss = 0
    total_rouge1 = 0
    total_rouge2 = 0
    total_rougeL = 0
    is_train = optimizer is not None
    model.train(is_train)
    name = name or ''
    batches_count = len(data_iter)
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    with torch.autograd.set_grad_enabled(is_train):
        with tqdm(total=batches_count) as progress_bar:
            for batch in data_iter:
                src, tgt, src_mask, tgt_mask = convert_batch(batch)
                enc_out = model.encoder(src, src_mask)

                bs, tgt_len = tgt.size()
                output_teacher = tgt[:, 0].unsqueeze(1)
                output_autoregressive = tgt[:, 0].unsqueeze(1)

                for t in range(1, tgt_len):
                    logits_teacher = model.decoder(output_teacher, enc_out, src_mask, tgt_mask[:, :t, :t])
                    logits_autoreg = model.decoder(output_autoregressive, enc_out, src_mask, tgt_mask[:, :t, :t])
                    next_token = logits_teacher[:, -1, :].argmax(dim=-1)
                    tf = random.random() < teacher_forcing_ratio
                    next_input_teacher = tgt[:, t] if tf else next_token
                    output_teacher = torch.cat((output_teacher, next_input_teacher.unsqueeze(1)), dim=1)
                    output_autoregressive = torch.cat((output_autoregressive, next_token.unsqueeze(1)), dim=1)

                logits = model.decoder(output_teacher[:, :-1], enc_out, src_mask, tgt_mask[:, :-1, :-1])
                logits = logits.contiguous().view(-1, logits.shape[-1])
                target = tgt[:, 1:].contiguous().view(-1)
                loss = criterion(logits, target)
                epoch_loss += loss.item()

                preds = decode_tokens(output_autoregressive, model.decoder._emb[0].weight)
                targets = decode_tokens(tgt[:, 1:], model.decoder._emb[0].weight)
                batch_r1 = batch_r2 = batch_rL = 0
                for pred, ref in zip(preds, targets):
                    scores = scorer.score(ref, pred)
                    batch_r1 += scores['rouge1'].fmeasure
                    batch_r2 += scores['rouge2'].fmeasure
                    batch_rL += scores['rougeL'].fmeasure
                total_rouge1 += batch_r1 / bs
                total_rouge2 += batch_r2 / bs
                total_rougeL += batch_rL / bs

                if optimizer:
                    optimizer.optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                progress_bar.update()
                progress_bar.set_description(f"{name} Loss={loss.item():.4f}")

    return {
        'loss': epoch_loss / batches_count,
        'ppx': math.exp(epoch_loss / batches_count),
        'rouge1': total_rouge1 / batches_count,
        'rouge2': total_rouge2 / batches_count,
        'rougeL': total_rougeL / batches_count,
    }


def fit(model, criterion, optimizer, train_iter, epochs_count=1, val_iter=None,
        save_path1=None, patience=7, start_epoch=0, best_val_loss=float('inf'),
        train_metrics_file=Path("train_metrics.csv"), val_metrics_file=Path("val_metrics.csv")):

    epochs_no_improve = 0
    best_model_state = None
    tf_start = 0.7
    tf_min = 0.1
    tf_decay = 0.03

    for epoch in range(start_epoch, start_epoch + epochs_count):
        tf_ratio = max(tf_min, tf_start - tf_decay * epoch)
        train_metrics = do_epoch(model, criterion, train_iter, optimizer, f"[Train {epoch+1}]", tf_ratio)
        _save_metrics(train_metrics_file, epoch, train_metrics)

        if val_iter is not None:
            val_metrics = do_epoch(model, criterion, val_iter, None, f"[Val {epoch+1}]", teacher_forcing_ratio=0.0)
            _save_metrics(val_metrics_file, epoch, val_metrics)
            val_loss = val_metrics['loss']

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                epochs_no_improve = 0
                if save_path1:
                    torch.save({
                        'model_state_dict': best_model_state,
                        'optimizer_state_dict': optimizer.optimizer.state_dict(),
                        'noam_step': optimizer._step,
                        'best_val_loss': best_val_loss,
                        'epoch': epoch
                    }, save_path1)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                    break


def _save_metrics(path: Path, epoch: int, metrics: dict):
    metrics['epoch'] = epoch
    df = pd.DataFrame([metrics])
    if not path.exists():
        df.to_csv(path, index=False)
    else:
        df.to_csv(path, mode='a', header=False, index=False)
