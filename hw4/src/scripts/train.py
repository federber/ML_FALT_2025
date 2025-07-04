
import argparse
import torch
from training.simclr_train import train_simclr
from training.protonet_train import train_protonet
from models import SimCLRModel, ProtoNet, Encoder
from data import CIFAR10Dataset, OmniglotDataset, simclr_transform, proto_transform

def main():
    parser = argparse.ArgumentParser(description='Train SimCLR/ProtoNet models')
    parser.add_argument('--model', type=str, required=True, choices=['simclr', 'protonet'],
                       help='Model to train: simclr or protonet')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == 'simclr':
        model = SimCLRModel().to(device)
        train_loader = CIFAR10Dataset.get_loader(
            train=True,
            transform=simclr_transform,
            batch_size=args.batch_size
        )
        train_simclr(model, train_loader, args.epochs, args.lr, device)

    elif args.model == 'protonet':
        encoder = Encoder().to(device)
        model = ProtoNet(encoder).to(device)
        train_loader = OmniglotDataset.get_loader(
            mode='train',
            transform=proto_transform,
            n_way=5,
            k_shot=5
        )
        train_protonet(model, train_loader, args.epochs, args.lr, device)

if __name__ == "__main__":
    main()