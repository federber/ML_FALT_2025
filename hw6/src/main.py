import os
import sys
from src.train import TrainingSession
from src.eval import EvaluationSession
from src.visualization import visualize_training

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Train new model: python dqn.py train [--model MODEL_PATH] [--log LOG_PATH]")
        print("  Test model: python dqn.py test MODEL_PATH")
        print("  Visualize training: python dqn.py visualize LOG_PATH")
        sys.exit(1)

    mode = sys.argv[1]

    if mode == 'train':
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', help='Path to existing model to continue training')
        parser.add_argument('--log', help='Path to training log file')
        args = parser.parse_args(sys.argv[2:])

        os.makedirs('models', exist_ok=True)
        session = TrainingSession(model_path=args.model, log_path=args.log)
        session.run()

    elif mode == 'test':
        if len(sys.argv) < 3:
            print("Please specify model path")
            sys.exit(1)

        evaluator = EvaluationSession(sys.argv[2])
        evaluator.run()

    elif mode == 'visualize':
        if len(sys.argv) < 3:
            print("Please specify log path")
            sys.exit(1)

        visualize_training(sys.argv[2])

    else:
        print("Invalid mode. Use 'train', 'test', or 'visualize'")


if __name__ == "__main__":
    main()
