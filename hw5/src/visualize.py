import matplotlib.pyplot as plt


def plot_loss(history):
    epochs = history["epoch"]
    train_loss = history["train_loss"]
    valid_loss = history["valid_loss"]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, valid_loss, label="Valid Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.grid()
    plt.legend()
    plt.show()


def plot_learning_rate(history):
    epochs = history["epoch"]
    lr = history["learning_rate"]

    plt.figure(figsize=(8, 3))
    plt.plot(epochs, lr, label="Learning Rate")
    plt.xlabel("Epoch")
    plt.ylabel("LR")
    plt.title("Learning Rate Schedule")
    plt.grid()
    plt.legend()
    plt.show()


def plot_roc(fpr, tpr):
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.grid()
    plt.show()


def plot_precision_threshold(thresholds, precision):
    plt.figure()
    plt.plot(thresholds, precision)
    plt.xlabel("Threshold")
    plt.ylabel("Precision")
    plt.title("Precision vs Threshold")
    plt.grid()
    plt.show()


def plot_recall_threshold(thresholds, recall):
    plt.figure()
    plt.plot(thresholds, recall)
    plt.xlabel("Threshold")
    plt.ylabel("Recall")
    plt.title("Recall vs Threshold")
    plt.grid()
    plt.show()


def plot_f1_threshold(thresholds, f1_score):
    plt.figure()
    plt.plot(thresholds, f1_score)
    plt.xlabel("Threshold")
    plt.ylabel("F1-Score")
    plt.title("F1-Score vs Threshold")
    plt.grid()
    plt.show()


def plot_fbeta_threshold(thresholds, f_beta):
    plt.figure()
    plt.plot(thresholds, f_beta)
    plt.xlabel("Threshold")
    plt.ylabel("F-beta Score")
    plt.title("F-beta vs Threshold")
    plt.grid()
    plt.show()