import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import pickle
from sklearn.manifold import TSNE
from dataset import SeedDataset
from models import create_model
from parser import parse_arguments
import torch
import pandas as pd

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
checkpoint_dir = "../checkpoint"
figure_dir = "../figure"

def plot_embedding(emb, label, title):
    fig = plt.figure(figsize=(4.5, 4.5))
    tsne = TSNE(perplexity=50, n_components=2, init='pca', early_exaggeration=12, learning_rate=1000, n_iter=3000)
    feature_2d = tsne.fit_transform(emb)
    x_min, x_max = np.min(feature_2d, 0), np.max(feature_2d, 0)
    feature_2d = (feature_2d - x_min) / (x_max - x_min)

    n_class = 2
    df = pd.DataFrame()
    df['pca-one'] = feature_2d[:, 0]
    df['pca-two'] = feature_2d[:, 1]
    df['label'] = label

    sns.scatterplot(
        x='pca-one', y='pca-two',
        hue='label',
        palette=sns.color_palette("hls", n_class),
        data=df,
        legend="full",
        alpha=0.7
    )

    plt.xlabel(""); plt.ylabel("")
    fig.tight_layout()
    fig.savefig(os.path.join(figure_dir, title+".pdf"))

def plot_subject_accs():
    fig = plt.figure(figsize=(9, 4))
    sns.set(style='whitegrid')

    xticks = np.arange(16)

    MLP_y = [0.6909, 0.6585, 0.6016, 0.7319, 0.632, 0.6532,	0.6405,	0.4643,	0.7139,	0.7065,	0.8005,	0.6037,	0.9582,	0.5628,	0.764, 0.6788]
    MLP_yerr = [0.] * 15 + [0.1098]
    DANN_y = [0.8226, 0.7702, 0.8265, 0.8869, 0.675, 0.6942, 0.8533, 0.6538, 0.8524, 0.8757, 0.8913, 0.731,	0.9753,	0.896, 0.8795, 0.8189]
    DANN_yerr = [0.] * 15 + [0.0906]
    ASDA_y = [0.8226, 0.7702, 0.8265, 0.8869, 0.675, 0.6942, 0.8533, 0.6538, 0.8524, 0.8757, 0.8913, 0.731, 0.9753, 0.896, 0.8795, 0.8189]
    ASDA_yerr = [0.] * 15 + [0.0906]

    plt.bar(x=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', 'mean'],
            height=MLP_y, yerr=MLP_yerr, color='tomato',
            label='MLP', alpha=1.0, width=0.27)
    plt.bar(x=xticks + 0.25, height=DANN_y, yerr=DANN_yerr, color='sienna', label='DANN', alpha=1.0, width=0.27)
    plt.bar(x=xticks + 0.5, height=ASDA_y, yerr=ASDA_yerr, color='purple', label='DANN', alpha=1.0, width=0.27)
    plt.xticks(xticks + 0.3)

    plt.ylabel('Accuracy', size=18)
    plt.xlabel('Subject Index', size=18)
    plt.legend(loc='upper center', fontsize=12, ncol=3, bbox_to_anchor=(0.5, 1.2))
    plt.tick_params(labelsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(figure_dir, 'acc.pdf'), bbox_inches='tight')

def plot_train_curves(gen_loss, dis_loss, mean_prob):
    sns.set_style('whitegrid')
    fig, ax = plt.subplots(1, 3, figsize=(12, 3))

    # plot generator loss curve
    ax[0].plot(gen_loss, linestyle='-', antialiased=True)
    ax[0].set_xlabel('Iteration', size=12)
    ax[0].set_ylabel('Generator Loss', size=12)

    # plot discriminator loss curve
    ax[1].plot(dis_loss, linestyle='-', antialiased=True)
    ax[1].set_xlabel('Iteration', size=12)
    ax[1].set_ylabel('Discriminator Loss', size=12)

    # plot mean prob curve
    ax[2].plot(mean_prob, linestyle='-', antialiased=True)
    ax[2].set_xlabel('Iteration', size=12)
    ax[2].set_ylabel('Mean Probability', size=12)

    fig.tight_layout()
    fig.savefig(os.path.join(figure_dir, "augmentation_curve.pdf"))

if __name__ == '__main__':
    args = parse_arguments()
    dataset = SeedDataset(False)

    dataset.prepare_dataset(0)
    source_x, source_y = torch.tensor(dataset.x).to(device).float(), dataset.y
    dataset.prepare_dataset(14)
    target_x, target_y = torch.tensor(dataset.x).to(device).float(), dataset.y

    model = create_model(args).to(device)
    model_state_dict = torch.load(os.path.join(checkpoint_dir, args.model+'_checkpoint.pt'))
    model.load_state_dict(model_state_dict)
    model.eval()

    source_feature = model.label_classifier(model.feature_extractor(source_x)).detach().cpu().numpy()
    target_feature = model.label_classifier(model.feature_extractor(target_x)).detach().cpu().numpy()
    feature_mapping = np.concatenate((source_feature, target_feature), axis=0)
    labels = ['source'] * source_feature.shape[0] + ['target'] * target_feature.shape[0]

    plot_embedding(feature_mapping, labels, args.model)

    # plot_subject_accs()

