import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch


def visualize_attention(attn_weights, source_tokens, target_tokens, layer=-1, head=0):
    """
    Визуализирует attention для заданного слоя и головы.

    Args:
        attn_weights (List[Tensor]): attention-матрицы из модели [layers][batch, heads, tgt_len, src_len]
        source_tokens (List[str]): токены входа
        target_tokens (List[str]): токены выхода
        layer (int): индекс слоя (по умолчанию последний)
        head (int): индекс attention-главы (по умолчанию 0)
    """
    if not isinstance(attn_weights, list):
        raise ValueError("attn_weights должно быть списком attention матриц по слоям")

    attn = attn_weights[layer][0, head].detach().cpu().numpy()  # [tgt_len, src_len]

    plt.figure(figsize=(10, 8))
    sns.heatmap(attn, xticklabels=source_tokens, yticklabels=target_tokens, cmap="viridis")
    plt.xlabel("Source")
    plt.ylabel("Target")
    plt.title(f"Attention: Layer {layer}, Head {head}")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
