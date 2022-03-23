from torch.autograd import Function
import torch.nn as nn
import torch
from torch import autograd
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

def create_model(args):
    if args.model == 'DANN':
        return DANN(310, args.hidden_dim, 3, 2, args.lamda)
    elif args.model == 'ASDA':
        return ASDA(310, args.hidden_dim, 3, 2, args.lamda, args.triplet_weight)
    elif args.model == 'MLP':
        return MLP(310, args.hidden_dim, 3)
    elif args.model == 'ResNet':
        return ResNet(310, args.hidden_dim, 3)
    elif args.model == 'IRM':
        return IRM(310, args.hidden_dim, 3, args.penalty_weight)
    elif args.model == 'REx':
        return REx(310, args.hidden_dim, 3, args.variance_weight)
    elif args.model == 'WGANGen':
        return WGANGen(64, 310, 128)
    elif args.model == 'ADDA':
        return ADDA(310, args.hidden_dim, 3, 1, 10)
    else:
        raise ValueError("Unknown model type!")

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_labels):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.criterion = nn.CrossEntropyLoss()

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.label_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, num_labels)
        )

    def forward(self, input_data):
        feature_mapping = self.feature_extractor(input_data)
        class_output = self.label_classifier(feature_mapping)
        return class_output

    def compute_loss(self, data):
        x, y = data
        class_output = self.forward(x)
        loss = self.criterion(class_output, y)
        return loss

class ResNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_labels):
        super(ResNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.criterion = nn.CrossEntropyLoss()

        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(input_dim, hidden_dim))
        for i in range(3):
            self.lins.append(nn.Linear(hidden_dim, hidden_dim))
        self.lins.append(nn.Linear(hidden_dim, num_labels))

    def forward(self, input_data):
        x = self.lins[0](input_data)
        x = F.relu(x, inplace=True)
        for i, lin in enumerate(self.lins[1:-1]):
            x_ = lin(x)
            x_ = F.relu(x_, inplace=True)
            x = x_ + x
        x = self.lins[-1](x)
        return x

    def compute_loss(self, data):
        x, y = data
        class_output = self.forward(x)
        loss = self.criterion(class_output, y)
        return loss

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, lamda):
        ctx.lamda = lamda
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lamda
        return output, None

class DANN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_labels, num_domains, lamda):
        super(DANN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.num_domains = num_domains
        self.lamda = lamda
        self.criterion = nn.CrossEntropyLoss()

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.label_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, num_labels)
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, num_domains)
        )

    def forward(self, input_data):
        feature_mapping = self.feature_extractor(input_data)
        reverse_feature = ReverseLayerF.apply(feature_mapping, self.lamda)
        class_output = self.label_classifier(feature_mapping)
        domain_output = self.domain_classifier(reverse_feature)
        return class_output, domain_output

    def compute_loss(self, source_data, target_data):
        inputs, class_labels, domain_source_labels = source_data
        pred_class_label, pred_domain_label = self.forward(inputs)
        class_loss = self.criterion(pred_class_label, class_labels)
        domain_source_loss = self.criterion(pred_domain_label, domain_source_labels)

        inputs, domain_target_labels = target_data
        _, pred_domain_label = self.forward(inputs)
        domain_target_loss = self.criterion(pred_domain_label, domain_target_labels)

        domain_loss = domain_source_loss + domain_target_loss
        return class_loss, domain_loss


class ADDA:
    def __init__(self, input_dim, hidden_dim, num_labels, num_domains, lamda):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.num_domains = num_domains
        self.lamda = lamda
        self.criterion = nn.CrossEntropyLoss()

        self.srcMapper = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.ReLU()
        )

        self.tgtMapper = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.ReLU()
        )

        self.Classifier = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_labels)
        )

        self.Discriminator = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_domains)
        )

    def pretrain_forward(self, input_data):
        src_mapping = self.srcMapper(input_data)
        src_class = self.Classifier(src_mapping)
        return src_class

    def pretrain_loss(self, train_data):
        x, y = train_data
        src_class = self.pretrain_forward(x)
        loss = self.criterion(src_class, y)
        return loss

    def discriminator_loss(self, src_data, tgt_data):
        src_x, src_y = src_data
        tgt_x, tgt_y = tgt_data
        src_mapping = self.srcMapper(src_x)
        tgt_mapping = self.tgtMapper(tgt_x)
        batch_size = src_mapping.size(0)
        alpha = torch.rand(batch_size, 1).to(src_mapping.device)
        alpha = alpha.expand(src_mapping.size())
        interpolates = alpha * src_mapping + (1 - alpha) * tgt_mapping
        interpolates = autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates = self.Discriminator(interpolates)
        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).to(interpolates.device),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        pred_tgt, pred_src = self.Discriminator(tgt_mapping), self.Discriminator(src_mapping)
        loss_discriminator = pred_tgt.mean() - pred_src.mean() + self.lamda * gradient_penalty
        return loss_discriminator

    def tgt_loss(self, tgt_data):
        tgt_x, tgt_y = tgt_data
        tgt_mapping = self.tgtMapper(tgt_x)
        pred_tgt = self.Discriminator(tgt_mapping)
        loss_tgt = -pred_tgt.mean()
        return loss_tgt


class IRM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_labels, penalty_weight):
        super(IRM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.penalty_weight = penalty_weight
        self.criterion = nn.CrossEntropyLoss()

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.label_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, num_labels)
        )

    def forward(self, input_data):
        feature_mapping = self.feature_extractor(input_data)
        class_output = self.label_classifier(feature_mapping)
        return class_output

    def penalty(self, logits, y):
        scale = torch.ones((1, self.num_labels)).to(y.device).requires_grad_()
        loss = self.criterion(logits * scale, y)
        grad = autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(grad ** 2)

    def compute_loss(self, data):
        x, y = data
        class_output = self.forward(x)
        loss = self.criterion(class_output, y)
        penalty = self.penalty(class_output, y)
        return loss + self.penalty_weight * penalty

class REx(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_labels, variance_weight):
        super(REx, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.variance_weight = variance_weight
        self.criterion = nn.CrossEntropyLoss(reduction='none')

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.label_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, num_labels)
        )

    def forward(self, input_data):
        feature_mapping = self.feature_extractor(input_data)
        class_output = self.label_classifier(feature_mapping)
        return class_output

    def compute_loss(self, data):
        x, y = data
        class_output = self.forward(x)
        loss = self.criterion(class_output, y)
        loss_mean = torch.mean(loss)
        loss_var = torch.var(loss)
        return loss_mean + self.variance_weight * loss_var

class WGANGen:
    def __init__(self, noise_dim, input_dim, hidden_dim):
        self.noise_dim = noise_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.netG = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, input_dim)
        )

        self.netD = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, 1)
        )

class ASDA(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_labels, num_domains, lamda, triplet_weight):
        super(ASDA, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.num_domains = num_domains
        self.lamda = lamda
        self.triplet_weight = triplet_weight
        self.criterion = nn.CrossEntropyLoss(reduction='none')

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.label_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, num_labels)
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, num_domains)
        )

    def forward(self, input_data):
        _, class_output, domain_output = self.record_forward(input_data)
        return class_output, domain_output

    def record_forward(self, input_data):
        feature_mapping = self.feature_extractor(input_data)
        reverse_feature = ReverseLayerF.apply(feature_mapping, self.lamda)
        class_output = self.label_classifier(feature_mapping)
        domain_output = self.domain_classifier(reverse_feature)
        return feature_mapping, class_output, domain_output

    def separability_loss(self, labels, latents, imbalance_parameter=1):
        criteria = nn.modules.loss.CosineEmbeddingLoss()
        loss_up = 0
        one_cuda = torch.ones(1).cuda()
        mean = torch.mean(latents, dim=0).cuda().view(1, -1)
        loss_down = 0
        for i in range(self.num_labels):
            indecies = labels.eq(i)
            mean_i = torch.mean(latents[indecies], dim=0).view(1, -1)
            if str(mean_i.norm().item()) != 'nan':
                for latent in latents[indecies]:
                    loss_up += criteria(latent.view(1, -1), mean_i, one_cuda)
                loss_down += criteria(mean, mean_i, one_cuda)
        loss = (loss_up / loss_down) * imbalance_parameter
        return loss

    def pseudo_labeling(self, pred_class_label, m=.8):
        pred_class_label = F.softmax(pred_class_label, dim=1)
        pred_class_prob, pred_class_label = torch.max(pred_class_label, dim=1)
        indices = pred_class_prob > m
        pseudo_label = pred_class_label[indices]
        _, counts = np.unique(pseudo_label.cpu().numpy(), return_counts=True)
        if counts.shape[0] == 0:
            return False
        else:
            mi = np.min(counts)
            if len(counts) < 10:
                mi = 0
            ma = np.max(counts)
            return indices, pseudo_label, (mi + 1) / (ma + 1)

    def compute_loss(self, source_data, target_data):
        source_inputs, source_labels, domain_source_labels = source_data
        target_inputs, domain_target_labels = target_data

        latent_source, pred_class_label, pred_domain_label = self.record_forward(source_inputs)
        source_class_loss = self.criterion(pred_class_label, source_labels).mean()
        source_entropy = Categorical(logits=pred_class_label).entropy()
        source_domain_loss = (
                    (torch.ones_like(source_entropy) + source_entropy.detach() / self.num_labels) * self.criterion(
                pred_domain_label, domain_source_labels)).mean()

        latent_target, pred_class_label, pred_domain_label = self.record_forward(target_inputs)
        target_entropy = Categorical(logits=pred_class_label).entropy()
        target_domain_loss = ((torch.ones_like(target_entropy) + target_entropy.detach() / self.num_labels) * self.criterion(pred_domain_label, domain_target_labels)).mean()

        sep_loss = 0.
        data = self.pseudo_labeling(pred_class_label)
        if data:
            indices, pseudo_labels, imbalance_parameter = data
            latent_target = latent_target[indices, :]
            sep_loss = self.separability_loss(torch.cat((source_labels, pseudo_labels)),
                                              torch.cat((latent_source, latent_target)),
                                              imbalance_parameter)

        return source_class_loss, (source_domain_loss + target_domain_loss) * self.lamda +  sep_loss * self.triplet_weight



