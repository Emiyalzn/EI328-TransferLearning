import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="General Traning Pipeline")
    parser.add_argument("--model", type=str, default='svm')
    parser.add_argument("--svm_c", type=float, default=.1)
    parser.add_argument("--svm_kernel", choices=['linear', 'rbf', 'sigmoid', 'poly'], default='linear')
    parser.add_argument("--lamda", type=float, default=0.5)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_epoch", type=int, default=300)
    parser.add_argument("--display_epoch", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()