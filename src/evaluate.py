import torch
import heapq
import torch.nn.functional as F

def clean_text(tokens):
    return " ".join([t for t in tokens if t not in {"<s>", "</s>", "<pad>", "<unk>"}])

def subsequent_mask(size, device):
    return torch.triu(torch.ones(size, size, device=device), diagonal=1).bool().unsqueeze(0)

def generate_summary_beam(model, src_sentence, src_field, tgt_field, device,
                           max_len=20, beam_width=5, length_penalty=0.6, no_repeat_ngram_size=2):
    model.eval()
    tokens = [src_field.init_token] + src_field.tokenize(src_sentence) + [src_field.eos_token]
    indexed = [src_field.vocab.stoi.get(t, src_field.vocab.stoi[src_field.unk_token]) for t in tokens]
    src_tensor = torch.LongTensor(indexed).unsqueeze(0).to(device)
    src_mask = (src_tensor != src_field.vocab.stoi[src_field.pad_token]).unsqueeze(1).unsqueeze(2)

    with torch.no_grad():
        enc_out = model.encoder(src_tensor, src_mask)

    beams = [(0.0, [tgt_field.vocab.stoi[tgt_field.init_token]])]

    for _ in range(max_len):
        new_beams = []
        for score, seq in beams:
            trg_tensor = torch.LongTensor(seq).unsqueeze(0).to(device)
            trg_mask = subsequent_mask(trg_tensor.size(1), device)
            with torch.no_grad():
                logits = model.decoder(trg_tensor, enc_out, src_mask, trg_mask)
                log_probs = F.log_softmax(logits[:, -1], dim=-1)

            topk_log_probs, topk_ids = log_probs.topk(beam_width)
            for log_prob, idx in zip(topk_log_probs[0], topk_ids[0]):
                new_seq = seq + [idx.item()]
                new_score = score + log_prob.item()
                new_beams.append((new_score, new_seq))

        beams = heapq.nlargest(beam_width, new_beams, key=lambda x: x[0] / (len(x[1]) ** length_penalty))
        if all(seq[-1] == tgt_field.vocab.stoi[tgt_field.eos_token] for _, seq in beams):
            break

    best_seq = max(beams, key=lambda x: x[0])[1]
    tokens = [tgt_field.vocab.itos[i] for i in best_seq]
    return clean_text(tokens[1:-1])