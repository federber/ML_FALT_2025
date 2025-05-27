import math
import random
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import pandas as pd
from pathlib import Path
from src.data import convert_batch

tqdm.get_lock().locks = []
from rouge_score import rouge_scorer



class NoamOpt(object):
    def __init__(self, model_size, factor=2, warmup=4000, optimizer=None):
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step = None):
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))


class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1, vocab_size=None, ignore_index=-100):
        super(LabelSmoothingLoss, self).__init__()
        assert 0.0 <= smoothing < 1.0
        self.smoothing = smoothing
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        # pred: (batch_size * seq_len, vocab_size)
        # target: (batch_size * seq_len)

        pred = pred.log_softmax(dim=-1)

        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.vocab_size - 2))
            mask = (target != self.ignore_index).unsqueeze(1)
            true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - self.smoothing)
            true_dist = true_dist * mask

        return torch.mean(torch.sum(-true_dist * pred, dim=-1))



def do_epoch(model, criterion, data_iter, optimizer=None, name=None, teacher_forcing_ratio=0.5):
    epoch_loss = 0
    total_rouge1 = 0
    total_rouge2 = 0
    total_rougeL = 0

    is_train = optimizer is not None
    name = name or ''
    model.train(is_train)

    batches_count = len(data_iter)

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    with torch.autograd.set_grad_enabled(is_train):
        with tqdm(total=batches_count) as progress_bar:
            for i, batch in enumerate(data_iter):
                source_inputs, target_inputs, source_mask, target_mask = convert_batch(batch)
                encoder_output = model.encoder(source_inputs, source_mask)

                batch_size, target_len = target_inputs.size()

                output_teacher = target_inputs[:, 0].unsqueeze(1)  # Для обучения (с Teacher Forcing)
                output_autoregressive = target_inputs[:, 0].unsqueeze(1)  # Для ROUGE (без Teacher Forcing)

                for t in range(1, target_len):
                    # Общая часть: вычисление logits для обоих режимов
                    logits_teacher = model.decoder(output_teacher, encoder_output, source_mask, target_mask[:, :t, :t])
                    logits_autoregressive = model.decoder(output_autoregressive, encoder_output, source_mask, target_mask[:, :t, :t])

                    next_token_logits = logits_teacher[:, -1, :]
                    next_token = next_token_logits.argmax(dim=-1)

                    # Teacher Forcing для обучения
                    teacher_force = random.random() < teacher_forcing_ratio
                    next_input_teacher = target_inputs[:, t] if teacher_force else next_token
                    next_input_teacher = next_input_teacher.unsqueeze(1)
                    output_teacher = torch.cat((output_teacher, next_input_teacher), dim=1)

                    # Авторегрессия для ROUGE
                    next_input_autoregressive = next_token
                    output_autoregressive = torch.cat((output_autoregressive, next_input_autoregressive.unsqueeze(1)), dim=1)

                # Вычисление потерь на предсказаниях с Teacher Forcing
                logits = model.decoder(output_teacher[:, :-1], encoder_output, source_mask, target_mask[:, :-1, :-1])
                logits = logits.contiguous().view(-1, logits.shape[-1])
                target = target_inputs[:, 1:].contiguous().view(-1)
                loss = criterion(logits, target)
                epoch_loss += loss.item()


                # Вычисление ROUGE на авторегрессионных предсказаниях
                pred_sentences = decode_tokens(output_autoregressive, vocab)
                target_sentences = decode_tokens(target_inputs[:, 1:], vocab)  # Игнорируем начальный токен


                # Считаем метрики Rouge
                batch_rouge1 = 0
                batch_rouge2 = 0
                batch_rougeL = 0
                for pred, ref in zip(pred_sentences, target_sentences):
                    scores = scorer.score(ref, pred)
                    batch_rouge1 += scores['rouge1'].fmeasure
                    batch_rouge2 += scores['rouge2'].fmeasure
                    batch_rougeL += scores['rougeL'].fmeasure

                total_rouge1 += batch_rouge1 / batch_size
                total_rouge2 += batch_rouge2 / batch_size
                total_rougeL += batch_rougeL / batch_size


                if optimizer:
                    optimizer.optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                progress_bar.update()
                progress_bar.set_description(
                    '{:>5s} Loss = {:.5f}, PPX = {:.2f}, R1 = {:.3f}, R2 = {:.3f}, RL = {:.3f}'.format(
                        name, loss.item(), math.exp(loss.item()),
                        batch_rouge1 / batch_size, batch_rouge2 / batch_size, batch_rougeL / batch_size
                    )
                )


            avg_loss = epoch_loss / batches_count
            avg_rouge1 = total_rouge1 / batches_count
            avg_rouge2 = total_rouge2 / batches_count
            avg_rougeL = total_rougeL / batches_count

            progress_bar.set_description(
                '{:>5s} Loss = {:.5f}, PPX = {:.2f}, R1 = {:.3f}, R2 = {:.3f}, RL = {:.3f}'.format(
                    name, avg_loss, math.exp(avg_loss), avg_rouge1, avg_rouge2, avg_rougeL
                )
            )
            progress_bar.refresh()

    return {
        'loss': avg_loss,
        'ppx': math.exp(avg_loss),
        'rouge1': avg_rouge1,
        'rouge2': avg_rouge2,
        'rougeL': avg_rougeL
    }

def fit(model, criterion, optimizer, train_iter, epochs_count=1, val_iter=None,
        save_path1=None, patience=7, start_epoch=0, best_val_loss=float('inf'),
        train_metrics_file = Path('train_metrics.csv'), val_metrics_file = Path('val_metrics.csv')):

    epochs_no_improve = 0
    best_model_state = None

    initial_teacher_forcing = 0.7
    min_teacher_forcing = 0.1
    decay_rate = 0.03

    for epoch in range(start_epoch, start_epoch + epochs_count):
        name_prefix = f"[{epoch + 1} / {start_epoch + epochs_count}] "
        teacher_forcing_ratio = max(min_teacher_forcing, initial_teacher_forcing - decay_rate * epoch)

        train_metrics = do_epoch(model, criterion, train_iter, optimizer, name_prefix + "Train:", teacher_forcing_ratio)
        train_loss = train_metrics['loss']
        save_metrics(train_metrics_file,epoch,train_metrics)
        if val_iter is not None:
            val_metrics = do_epoch(model, criterion, val_iter, None, name_prefix + "Val:", teacher_forcing_ratio=0.0)
            val_loss = val_metrics['loss']
            save_metrics(val_metrics_file,epoch,val_metrics)



            if val_loss < best_val_loss - 1e-4:  # небольшая дельта, чтобы избежать флуктуаций
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                epochs_no_improve = 0
                print(f"New best model found! val_loss={val_loss:.4f}")
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
                print(f"No improvement. Patience: {epochs_no_improve}/{patience}")
                if epochs_no_improve >= patience:
                    print("Early stopping triggered!")
                    if save_path and best_model_state is not None:
                        model.load_state_dict(best_model_state)
                    break

        print(f"Epoch {epoch + 1}: Teacher Forcing Ratio = {teacher_forcing_ratio:.2f}")


def decode_tokens(tensor, vocab, skip_tokens=('<pad>', '<sos>', '<eos>')):
    sentences = []
    skip_ids = [vocab.stoi[token] for token in skip_tokens if token in vocab.stoi]

    for seq in tensor:
        tokens = [
            vocab.itos[token.item()]
            for token in seq
            if token.item() not in skip_ids
        ]
        sentences.append(" ".join(tokens))

    return sentences


def save_metrics(save_path, epoch, metrics):
    metrics['epoch'] = epoch
    df = pd.DataFrame([metrics])

    if not save_path.exists():
        df.to_csv(save_path, index=False)
    else:
        df.to_csv(save_path, mode='a', header=False, index=False)