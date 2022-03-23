import torch
from torch.utils.data import DataLoader
import numpy as np
import os
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from dataset import SeedDataset
from models import create_model
from parser import parse_arguments
from utils import plot_embedding

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
checkpoint_dir = "../checkpoint"

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
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

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
                    filename = f"{args.model}_checkpoint.pt"
                    torch.save(model.state_dict(), os.path.join(checkpoint_dir, filename))
                print(f"Epoch {epoch}, Class Loss {total_class_loss / len(train_loader):.4f}, Domain Loss {total_domain_loss / len(train_loader):.4f}, Acc {test_acc:.4f}")
        validation_accs[idx] = best_acc
        print(f"Fold {idx} best acc: {validation_accs[idx]:.4f}")
    acc_mean = np.mean(validation_accs)
    acc_std = np.std(validation_accs)
    print(f"Average acc is: {acc_mean:.4f}±{acc_std:.4f}")

def train_generalization(args, train_dataset, test_dataset):
    validation_accs = np.zeros(15)
    for idx in range(15):
        train_dataset.prepare_dataset(idx)
        test_dataset.prepare_dataset(idx)
        test_x, test_y = torch.tensor(test_dataset.x).to(device), torch.tensor(test_dataset.y).to(device)

        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

        model = create_model(args).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay=args.weight_decay)

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
                    filename = f"{args.model}_checkpoint.pt"
                    torch.save(model.state_dict(), os.path.join(checkpoint_dir, filename))
                print(f"Epoch {epoch}, Loss {total_loss / len(train_loader):.4f}, Acc {test_acc:.4f}")
        validation_accs[idx] = best_acc
        print(f"Fold {idx} best acc: {validation_accs[idx]:.4f}")
    acc_mean = np.mean(validation_accs)
    acc_std = np.std(validation_accs)
    print(f"Average acc is: {acc_mean:.4f}±{acc_std:.4f}")

def train_adaptation(args, train_dataset, test_dataset):
    validation_accs = np.zeros(15)
    for idx in range(15):
        # pretrain
        print("start pretraining:")
        train_dataset.prepare_dataset(idx)
        test_dataset.prepare_dataset(idx)
        train_x, train_y = torch.tensor(train_dataset.x).to(device), torch.tensor(train_dataset.y).to(device)
        test_x, test_y = torch.tensor(test_dataset.x).to(device), torch.tensor(test_dataset.y).to(device)
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True)

        model = create_model(args)
        model.srcMapper.to(device);
        model.Classifier.to(device);
        model.tgtMapper.to(device);
        model.Discriminator.to(device)

        optimizer_src = torch.optim.Adam(model.srcMapper.parameters(), lr=args.lr, betas=(0.5, 0.9))
        optimizer_cls = torch.optim.Adam(model.Classifier.parameters(), lr=args.lr, betas=(0.5, 0.9))

        best_acc = 0.
        for epoch in range(args.pretrain_epoch):
            model.srcMapper.train()
            model.Classifier.train()
            total_loss = 0
            for i, data in enumerate(train_loader):
                x, y = data
                x, y = x.to(device).float(), y.to(device)
                optimizer_src.zero_grad()
                optimizer_cls.zero_grad()
                loss = model.pretrain_loss((x, y))
                loss.backward()
                optimizer_src.step()
                optimizer_cls.step()
                total_loss += loss.detach().cpu().numpy()
            if epoch % args.display_epoch == 0:
                model.srcMapper.eval()
                model.Classifier.eval()
                _, test_y_pred = torch.max(model.Classifier(model.srcMapper(test_x.float())), dim=1)
                _, train_y_pred = torch.max(model.Classifier(model.srcMapper(train_x.float())), dim=1)

                test_acc = (test_y_pred == test_y + 1).sum().item() / len(test_dataset)
                train_acc = (train_y_pred == train_y + 1).sum().item() / len(train_dataset)

                if train_acc > best_acc:
                    best_acc = train_acc
                    filename = f"{args.model}_srcMapper_checkpoint.pt"
                    torch.save(model.srcMapper.state_dict(), os.path.join(checkpoint_dir, filename))
                    filename = f"{args.model}_Classifier_checkpoint.pt"
                    torch.save(model.Classifier.state_dict(), os.path.join(checkpoint_dir, filename))
                print(
                    f"Epoch {epoch}, Loss {total_loss / len(train_loader):.4f}, Acc {test_acc:.4f}, Acc_train {train_acc:.4f}")

        # adversarial train
        print("start adversarial training:")

        # optimizer of tgtMapper and Discriminator
        # optimizer_tgt = torch.optim.Adam(model.tgtMapper.parameters(), lr = 1e-5, betas=(0.5,0.9))
        # optimizer_disc = torch.optim.Adam(model.Discriminator.parameters(), lr = 1e-5, betas=(0.5,0.9))
        optimizer_tgt = torch.optim.RMSprop(model.tgtMapper.parameters(), lr=1e-5)
        optimizer_disc = torch.optim.RMSprop(model.Discriminator.parameters(), lr=1e-5)
        train_iter = iter(train_loader)
        test_iter = iter(test_loader)

        for param in model.srcMapper.parameters():
            param.requires_grad = False
        for param in model.Classifier.parameters():
            param.requires_grad = False

        best_acc = 0.
        for iteration in range(args.advtrain_iteration):
            model.tgtMapper.train()
            model.Discriminator.train()
            total_disc_loss = 0
            total_target_loss = 0

            for p in model.Discriminator.parameters():
                p.requires_grad = True

            # train discriminator
            for _ in range(args.critic_iters):
                try:
                    src_x, src_y = next(train_iter)
                    if src_x.size(0) < args.batch_size:
                        train_iter = iter(train_loader)
                        src_x, src_y = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    src_x, src_y = next(train_iter)
                src_x, src_y = src_x.to(device).float(), src_y.to(device)

                try:
                    tgt_x, tgt_y = next(test_iter)
                    if tgt_x.size(0) < args.batch_size:
                        test_iter = iter(test_loader)
                        tgt_x, tgt_y = next(test_iter)
                except StopIteration:
                    test_iter = iter(test_loader)
                    tgt_x, tgt_y = next(test_iter)
                tgt_x, tgt_y = tgt_x.to(device).float(), tgt_y.to(device)

                optimizer_disc.zero_grad()
                loss_discriminator = model.discriminator_loss((src_x, src_y), (tgt_x, tgt_y))
                loss_discriminator.backward()
                optimizer_disc.step()

                total_disc_loss += loss_discriminator.detach().cpu().numpy() / args.critic_iters

            # train target feature extractor
            for p in model.Discriminator.parameters():
                p.requires_grad = False

            optimizer_disc.zero_grad()
            optimizer_tgt.zero_grad()
            loss_tgt = model.tgt_loss((tgt_x, tgt_y))
            loss_tgt.backward()
            optimizer_tgt.step()

            total_target_loss += loss_tgt.detach().cpu().numpy()

            model.tgtMapper.eval()
            model.Discriminator.eval()
            _, test_y_pred = torch.max(model.Classifier(model.tgtMapper(test_x.float())), dim=1)
            test_acc = (test_y_pred == test_y + 1).sum().item() / len(test_dataset)

            if test_acc > best_acc:
                best_acc = test_acc
                filename = f"{args.model}_tgtMapper_checkpoint.pt"
                torch.save(model.tgtMapper.state_dict(), os.path.join(checkpoint_dir, filename))
                filename = f"{args.model}_Discriminator_checkpoint.pt"
                torch.save(model.Discriminator.state_dict(), os.path.join(checkpoint_dir, filename))
            print(
                f"Iteration {iteration}, Discriminator Loss {total_disc_loss:.4f}, Target Loss {total_target_loss:.4f}, Acc {test_acc:.4f}")

        validation_accs[idx] = best_acc
        print(f"Fold {idx} best acc: {validation_accs[idx]:.4f}")

    acc_mean = np.mean(validation_accs)
    acc_std = np.std(validation_accs)
    print(f"Average acc is: {acc_mean:.4f}±{acc_std:.4f}")

if __name__ == '__main__':
    args = parse_arguments()
    train_dataset = SeedDataset(True, args.is_augmentation)
    test_dataset = SeedDataset(False, args.is_augmentation)

    # fix random seed for reproducibility
    if args.seed != None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        torch.backends.cudnn.deterministic = True

    if args.model == 'SVM':
        train_svm(args, train_dataset, test_dataset)
    elif args.model == 'DANN' or args.model == 'ASDA':
        train_DANN(args, train_dataset, test_dataset)
    elif args.model == 'ADDA':
        train_adaptation(args, train_dataset, test_dataset)
    elif args.model == 'MLP' or args.model == 'ResNet' or args.model == 'IRM' or args.model == 'REx':
        train_generalization(args, train_dataset, test_dataset)
    else:
        raise ValueError("Unknown model type!")
