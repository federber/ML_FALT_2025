
import torch

def evaluate(model, loader, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)

            total += y.size(0)
            correct += (predicted == y).sum().item()

    return correct / total

def evaluate_protonet(model, n_way=5, k_shot=5, test_episodes=100):
    model.eval()
    total_correct = 0

    with torch.no_grad():
        for _ in range(test_episodes):
            support, query = model.sample_episode(n_way, k_shot)

            prototypes = model.get_prototypes(support)

            logits = model(query, prototypes)
            _, predicted = torch.max(logits, 1)

            labels = torch.arange(n_way).repeat_interleave(k_shot).to(model.device)
            total_correct += (predicted == labels).sum().item()

    return total_correct / (n_way * k_shot * test_episodes)