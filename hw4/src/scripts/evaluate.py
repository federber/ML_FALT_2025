
import argparse
import torch
from models import SimCLRModel, ProtoNet, Encoder
from utils import evaluate_linear, evaluate_protonet
from data import CIFAR10Dataset, OmniglotDataset, clf_transform

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained models')
    parser.add_argument('--model', type=str, required=True, choices=['simclr', 'protonet'])
    parser.add_argument('--model_path', type=str, required=True, help='Path to saved model')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == 'simclr':
        model = SimCLRModel().to(device)
        model.load_state_dict(torch.load(args.model_path))
        test_loader = CIFAR10Dataset.get_loader(
            train=False,
            transform=clf_transform,
            batch_size=128
        )
        acc = evaluate_linear(model.encoder, test_loader, device)
        print(f"Linear Evaluation Accuracy: {acc:.4f}")

    elif args.model == 'protonet':
        encoder = Encoder().to(device)
        model = ProtoNet(encoder).to(device)
        model.load_state_dict(torch.load(args.model_path))
        test_loader = OmniglotDataset.get_loader(
            mode='test',
            transform=proto_transform,
            n_way=5,
            k_shot=5
        )
        acc = evaluate_protonet(model, test_loader, 5, 5, 100, device)
        print(f"5-way 5-shot Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()