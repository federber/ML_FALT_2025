import torch
import heapq
import torch.nn.functional as F

def clean_text(tokens):
    return " ".join([tok for tok in tokens if tok not in {"<s>", "</s>", "<pad>", "<unk>"}])

def generate_summary_beam(
    model, src_sentence, src_field, tgt_field, device,
    max_len=20, beam_width=5, length_penalty=0.6, no_repeat_ngram_size=2
):
    model.eval()

    tokens = [src_field.init_token] + src_field.tokenize(src_sentence) + [src_field.eos_token]
    indexed_tokens = [
        src_field.vocab.stoi.get(token, src_field.vocab.stoi[src_field.unk_token])
        for token in tokens
    ]
    src_tensor = torch.LongTensor(indexed_tokens).unsqueeze(0).to(device)
    src_mask = (src_tensor != src_field.vocab.stoi[src_field.pad_token]).unsqueeze(1).unsqueeze(2)

    with torch.no_grad():
        encoder_outputs = model.encoder(src_tensor, src_mask)

    beams = [(0.0, [tgt_field.vocab.stoi[tgt_field.init_token]])]

    for step in range(max_len):
        new_beams = []
        for score, seq in beams:
            trg_tensor = torch.LongTensor(seq).unsqueeze(0).to(device)
            trg_mask = (trg_tensor != tgt_field.vocab.stoi[tgt_field.pad_token]).unsqueeze(1).unsqueeze(2)
            trg_mask = trg_mask & subsequent_mask(trg_tensor.size(1)).to(device)

            with torch.no_grad():
                logits = model.decoder(trg_tensor, encoder_outputs, src_mask, trg_mask)
                log_probs = F.log_softmax(logits[:, -1], dim=-1)

            if no_repeat_ngram_size > 0 and len(seq) >= no_repeat_ngram_size:
                ngram = tuple(seq[-(no_repeat_ngram_size - 1):])
                banned = set()
                for i in range(len(seq) - no_repeat_ngram_size + 1):
                    if tuple(seq[i:i + no_repeat_ngram_size - 1]) == ngram:
                        banned.add(seq[i + no_repeat_ngram_size - 1])
                for token_id in banned:
                    log_probs[0, token_id] = -1e9

            if step < 5:
                eos_idx = tgt_field.vocab.stoi[tgt_field.eos_token]
                log_probs[0, eos_idx] -= 1.0

            topk_log_probs, topk_idxs = log_probs.topk(beam_width)

            for log_prob, idx in zip(topk_log_probs[0], topk_idxs[0]):
                token = idx.item()
                new_seq = seq + [token]
                new_score = score + log_prob.item()
                new_beams.append((new_score, new_seq))

        beams = heapq.nlargest(
            beam_width,
            new_beams,
            key=lambda x: x[0] / ((len(x[1]) ** length_penalty) if length_penalty > 0 else 1)
        )

        if all(seq[-1] == tgt_field.vocab.stoi[tgt_field.eos_token] for _, seq in beams):
            break

    best_seq = max(beams, key=lambda x: x[0])[1]
    tokens = [tgt_field.vocab.itos[i] for i in best_seq]

    return clean_text(tokens[1:-1])
def evaluate_model(model, word_field, val_iter):
    model.eval()
    pad_idx = word_field.vocab.stoi["<pad>"]

    criterion = LabelSmoothingLoss(
        smoothing=0.05,
        vocab_size=len(word_field.vocab),
        ignore_index=pad_idx
    ).to(DEVICE)

    val_metrics = do_epoch(model, criterion, val_iter, None, "Evaluation:", teacher_forcing_ratio=0.0)
    return val_metrics
