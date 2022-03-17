from parser import parse_arguments
import numpy as np
from torch.utils.data import DataLoader
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from dataset import SeedDataset

def train_svm(args, train_dataset, test_dataset):
    validation_accs = np.zeros(15)
    for idx in range(15):
        train_dataset.prepare_dataset(idx)
        test_dataset.prepare_dataset(idx)
        clf = make_pipeline(StandardScaler(), SVC(C=args.svm_c, kernel=args.svm_kernel, gamma='auto'))
        clf.fit(train_dataset.x, np.squeeze(train_dataset.y))
        test_predict = clf.predict(test_dataset.x)
        # print(classification_report(np.squeeze(test_dataset.y), test_predict))
        validation_accs[idx] = accuracy_score(np.squeeze(test_dataset.y), test_predict)
        print(f"Fold {idx} acc: {validation_accs[idx]:.4f}")
    acc_mean = np.mean(validation_accs)
    acc_std = np.std(validation_accs)
    print(f"Average acc is: {acc_mean:.4f}±{acc_std:.4f}")

if __name__ == '__main__':
    train_dataset = SeedDataset(True)
    test_dataset = SeedDataset(False)
    args = parse_arguments()
    if args.model == 'svm':
        train_svm(args, train_dataset, test_dataset)
