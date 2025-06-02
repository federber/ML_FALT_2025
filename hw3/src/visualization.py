import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import pandas as pd


class AttentionData():
    def __init__(self, model, word_field, inp_text, out_text):
        self.enc_self_attn = model.encoder.attn_probs
        self.dec_self_attn = model.decoder.self_attn_probs
        self.dec_enc_attn = model.decoder.enc_attn_probs
        self.tokens_inp = [word_field.init_token] + word_field.tokenize(inp_text.lower()) + [word_field.eos_token]
        self.tokens_out = [word_field.init_token] + word_field.tokenize(out_text.lower()) + [word_field.eos_token]

    def visualize(self, layer, head, mode=0, save_path=None):
        if mode == 0:
            arr = self.enc_self_attn
            x_tokens = self.tokens_inp
            y_tokens = self.tokens_inp
            title = "enc_self_attn"
        elif mode == 1:
            arr = self.dec_self_attn
            x_tokens = self.tokens_out
            y_tokens = self.tokens_out
            title = "dec_self_attn"
        elif mode == 2:
            arr = self.dec_enc_attn
            x_tokens = self.tokens_inp
            y_tokens = self.tokens_out
            title = "dec_enc_attn"
        else:
            raise ValueError("Invalid mode. Use 0, 1, or 2.")

        attn_tensor = arr[layer][0][0]

        attn_map = attn_tensor[head].cpu().numpy()
        if attn_map.ndim != 2:
            raise ValueError(f"Expected 2D attention map, got shape: {attn_map.shape}")

        # Plotting
        plt.figure(figsize=(10, 8))
        sns.heatmap(attn_map, xticklabels=x_tokens, yticklabels=y_tokens, cmap='viridis')
        plt.title(f"Attention Weights for {title} (Layer {layer}, Head {head})")
        plt.xlabel("Input Tokens")
        plt.ylabel("Output Tokens")

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()


def process_and_visualize_attention(custom_texts, model, word_field, generate_summary_beam, DEVICE, MAX_LAYERS, MAX_HEADS, output_dir='attention_maps'):
    os.makedirs(output_dir, exist_ok=True)

    for i, text in enumerate(custom_texts):
        summary = generate_summary_beam(model, text, word_field, word_field, DEVICE)
        print(f"=== [{i}] ===")
        print("Text:", text)
        print("Generated summary:", summary)
        print()

        ad = AttentionData(model, word_field, text, summary)
        text_dir = os.path.join(output_dir, f"text_{i}")
        os.makedirs(text_dir, exist_ok=True)

        for mode in range(3):  # 0 = enc_self_attn, 1 = dec_self_attn, 2 = dec_enc_attn
            for layer in range(MAX_LAYERS):
                for head in range(MAX_HEADS):
                    filename = f"text_{i}_mode_{mode}_head_{head}_layer_{layer}.png"
                    save_path = os.path.join(text_dir, filename)
                    ad.visualize(layer=layer, head=head, mode=mode, save_path=save_path)


def analyze_vocabulary(vocab, data=None, max_words_to_print=20):
    """
    Анализирует словарь и предоставляет информацию о его составе и качестве.

    Args:
        vocab: Объект словаря (например, word_field.vocab из torchtext).
        data:  Список или итератор с текстовыми данными (для оценки покрытия словаря).  Если None, оценка покрытия не производится.
        max_words_to_print: Максимальное количество самых частых и самых редких слов для отображения.

    Returns:
        None.  Функция печатает результаты анализа в консоль.
    """

    vocab_size = len(vocab)
    unk_index = vocab.stoi.get("<unk>")
    pad_index = vocab.stoi.get("<pad>")
    bos_index = vocab.stoi.get("<s>")
    eos_index = vocab.stoi.get("</s>")

    print(f"Размер словаря: {vocab_size}")

    print("\nПроверка специальных токенов:")
    if "<unk>" in vocab.stoi:
        print(f"  Токен <unk> найден в словаре, индекс: {unk_index}")
    else:
        print("  Токен <unk> НЕ найден в словаре.")
    if "<pad>" in vocab.stoi:
        print(f"  Токен <pad> найден в словаре, индекс: {pad_index}")
    else:
        print("  Токен <pad> НЕ найден в словаре.")
    if "<s>" in vocab.stoi:
        print(f"  Токен <s> (BOS) найден в словаре, индекс: {bos_index}")
    else:
        print("  Токен <s> (BOS) НЕ найден в словаре.")
    if "</s>" in vocab.stoi:
        print(f"  Токен </s> (EOS) найден в словаре, индекс: {eos_index}")
    else:
        print("  Токен </s> (EOS) НЕ найден в словаре.")

    if hasattr(vocab, 'freqs'):
        print("\nАнализ частотности слов:")
        vocab_words_sorted = sorted(vocab.freqs.items(), key=lambda x: -x[1])

        print(f"  {max_words_to_print} самых частых слов:")
        for i, (word, freq) in enumerate(vocab_words_sorted[:max_words_to_print]):
            print(f"    {i+1}: {word} — {freq}")

        print(f"\n  {max_words_to_print} самых редких слов:")
        for i, (word, freq) in enumerate(vocab_words_sorted[-max_words_to_print:]):
            print(f"    {vocab_size-max_words_to_print+i+1}: {word} — {freq}")
    else:
        print("\nИнформация о частотах слов недоступна в словаре.")

    if data:
        print("\nОценка покрытия словаря:")
        unk_count = 0
        total_tokens = 0
        for text in data:
            for token in text:
                total_tokens += 1
                if vocab.stoi.get(token, unk_index) == unk_index:
                    unk_count += 1

        unk_percentage = (unk_count / total_tokens) * 100
        print(f"  Всего токенов в данных: {total_tokens}")
        print(f"  Количество токенов <unk>: {unk_count}")
        print(f"  Процент токенов <unk>: {unk_percentage:.2f}%")

        if hasattr(vocab, 'freqs'):
            frequencies = [freq for word, freq in vocab_words_sorted]
            plt.figure(figsize=(10, 5))
            plt.plot(frequencies)
            plt.xlabel("Ранг слова")
            plt.ylabel("Частота")
            plt.title("Распределение частот слов")
            plt.yscale('log')
            plt.show()

        print("\nКритерии 'хорошести' словаря (ориентировочные):")
        if vocab_size < 1000:
            print("  Словарь слишком маленький. Рассмотрите увеличение min_freq или использование большего объема данных.")
        elif vocab_size > 50000:
            print("  Словарь может быть слишком большим.  Рассмотрите уменьшение объема данных, увеличение min_freq или использование subword токенизации.")

        if unk_percentage > 5.0:
            print("  Процент неизвестных слов слишком высок.  Рассмотрите уменьшение min_freq, использование большего объема данных или subword токенизацию.")
        else:
            print("  Процент неизвестных слов находится в приемлемом диапазоне.")
    else:
        print("\nДанные для оценки покрытия словаря не предоставлены.")


def plot_metrics(train_metrics_file,
                 val_metrics_file,
                 save_path='training_plots.png'):
    train_df = pd.read_csv(train_metrics_file)
    val_df = pd.read_csv(val_metrics_file)

    metrics = [col for col in train_df.columns if col != 'epoch']

    n_metrics = len(metrics)
    n_cols = 2
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    fig.suptitle('Training vs Validation Metrics', fontsize=16)

    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]

        ax.plot(train_df['epoch'], train_df[metric], label='Train', color='blue')
        ax.plot(val_df['epoch'], val_df[metric], label='Validation', color='orange')

        ax.set_title(metric)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid(True)

    if n_metrics % n_cols != 0:
        for j in range(i+1, n_rows*n_cols):
            fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
