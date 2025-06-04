import pandas as pd
import matplotlib.pyplot as plt


def visualize_training(path):
    data = pd.read_csv(path)

    plt.figure(figsize=(20, 5))
    plt.plot(data['epoch'], data['max_q'])
    plt.title('Maximum Q-values')
    plt.xlabel('Epoch')
    plt.ylabel('Q-value')
    plt.savefig('Q_val_plot.png')
    plt.close()

    plt.figure(figsize=(15, 5))
    plt.plot(data['epoch'], data['loss'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.savefig('loss_plot.png')
    plt.close()