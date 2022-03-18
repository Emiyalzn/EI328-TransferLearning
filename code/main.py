from parser import parse_arguments
import numpy as np
from torch.utils.data import DataLoader
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from dataset import SeedDataset
from models import create_model
import torch.nn as nn
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train_svm(args, train_dataset, test_dataset):
    validation_accs = np.zeros(15)
    for idx in range(15):
        train_dataset.prepare_dataset(idx)
        test_dataset.prepare_dataset(idx)
        clf = make_pipeline(StandardScaler(), SVC(C=args.svm_c, kernel=args.svm_kernel, gamma='auto'))
        clf.fit(train_dataset.x, train_dataset.y)
        test_predict = clf.predict(test_dataset.x)
        # print(classification_report(np.squeeze(test_dataset.y), test_predict))
        validation_accs[idx] = accuracy_score(test_dataset.y, test_predict)
        print(f"Fold {idx} acc: {validation_accs[idx]:.4f}")
    acc_mean = np.mean(validation_accs)
    acc_std = np.std(validation_accs)
    print(f"Average acc is: {acc_mean:.4f}±{acc_std:.4f}")

def train_DANN(args, train_dataset, test_datastet):
    validation_accs = np.zeros(15)
    for idx in range(15):
        train_dataset.prepare_dataset(idx)
        test_dataset.prepare_dataset(idx)
        test_x, test_y = torch.tensor(test_dataset.x).to(device), torch.tensor(test_dataset.y).to(device)

        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_datastet, batch_size=args.batch_size, shuffle=True)

        model = create_model(args).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        test_iter = iter(test_loader)
        best_acc = 0.
        for epoch in range(args.num_epoch):
            model.train()
            total_class_loss = 0.
            total_domain_loss = 0.
            for i, data in enumerate(train_loader):
                inputs, class_labels = data
                domain_source_labels = torch.zeros(len(inputs)).long()
                train_data = inputs.to(device).float(), class_labels.to(device), domain_source_labels.to(device)

                try:
                    inputs, _ = next(test_iter)
                except StopIteration:
                    test_iter = iter(test_loader)
                    inputs, _ = next(test_iter)

                domain_target_labels = torch.ones(len(inputs)).long()
                test_data = inputs.to(device).float(), domain_target_labels.to(device)

                class_loss, domain_loss = model.compute_loss(train_data, test_data)
                loss = class_loss + domain_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_class_loss += class_loss.detach().cpu().numpy()
                total_domain_loss += domain_loss.detach().cpu().numpy()
            if epoch % args.display_epoch == 0:
                model.eval()
                pred_class_label, _ = model(test_x.float())
                _, test_y_pred = torch.max(pred_class_label, dim=1)
                test_acc = (test_y_pred == test_y + 1).sum().item() / len(test_dataset)
                if test_acc > best_acc:
                    best_acc = test_acc
                print(f"Epoch {epoch}, Class Loss {total_class_loss / len(train_loader):.4f}, Domain Loss {total_domain_loss / len(train_loader):.4f}, Acc {test_acc:.4f}")
        validation_accs[idx] = best_acc
        print(f"Fold {idx} best acc: {validation_accs[idx]:.4f}")
    acc_mean = np.mean(validation_accs)
    acc_std = np.std(validation_accs)
    print(f"Average acc is: {acc_mean:.4f}±{acc_std:.4f}")

def train_adaptation(args, train_dataset, test_dataset):
    pass

def train_generalization(args, train_dataset, test_dataset):
    validation_accs = np.zeros(15)
    for idx in range(15):
        train_dataset.prepare_dataset(idx)
        test_dataset.prepare_dataset(idx)
        test_x, test_y = torch.tensor(test_dataset.x).to(device), torch.tensor(test_dataset.y).to(device)

        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

        model = create_model(args).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

        best_acc = 0.
        for epoch in range(args.num_epoch):
            model.train()
            total_loss = 0.
            for i, data in enumerate(train_loader):
                x, y = data
                x, y = x.to(device).float(), y.to(device)
                loss = model.compute_loss((x, y))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.detach().cpu().numpy()
            if epoch % args.display_epoch == 0:
                model.eval()
                _, test_y_pred = torch.max(model(test_x.float()), dim=1)
                test_acc = (test_y_pred == test_y + 1).sum().item()/len(test_dataset)
                if test_acc > best_acc:
                    best_acc = test_acc
                print(f"Epoch {epoch}, Loss {total_loss / len(train_loader):.4f}, Acc {test_acc:.4f}")
        validation_accs[idx] = best_acc
        print(f"Fold {idx} best acc: {validation_accs[idx]:.4f}")
    acc_mean = np.mean(validation_accs)
    acc_std = np.std(validation_accs)
    print(f"Average acc is: {acc_mean:.4f}±{acc_std:.4f}")


if __name__ == '__main__':
    train_dataset = SeedDataset(True)
    test_dataset = SeedDataset(False)

    args = parse_arguments()

    # fix random seed for reproducibility
    if args.seed != None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        torch.backends.cudnn.deterministic = True

    if args.model == 'svm':
        train_svm(args, train_dataset, test_dataset)
    elif args.model == 'DANN':
        train_DANN(args, train_dataset, test_dataset)
    elif args.model == 'ADDA' or args.model == 'WGANDA':
        train_adaptation(args, train_dataset, test_dataset)
    elif args.model == 'MLP' or args.model == 'IRM' or args.model == 'REx':
        train_generalization(args, train_dataset, test_dataset)
    else:
        raise ValueError("Unknown model type!")
