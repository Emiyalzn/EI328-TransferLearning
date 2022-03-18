from torch.autograd import Function
import torch.nn as nn
import torch
from torch import autograd
import torch.nn.functional as F

def create_model(args):
    if args.model == 'DANN':
        return DANN(310, args.hidden_dim, 3, 2, args.lamda)
    elif args.model == 'MLP':
        return MLP(310, args.hidden_dim, 3)
    elif args.model == 'ResNet':
        return ResNet(310, args.hidden_dim, 3)
    elif args.model == 'IRM':
        return IRM(310, args.hidden_dim, 3, args.penalty_weight)
    elif args.model == 'REx':
        return REx(310, args.hidden_dim, 3)

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

    def compute_loss(self, train_data, test_data):
        inputs, class_labels, domain_source_labels = train_data
        pred_class_label, pred_domain_label = self.forward(inputs)
        class_loss = self.criterion(pred_class_label, class_labels)
        domain_source_loss = self.criterion(pred_domain_label, domain_source_labels)

        inputs, domain_target_labels = test_data
        _, pred_domain_label = self.forward(inputs)
        domain_target_loss = self.criterion(pred_domain_label, domain_target_labels)

        domain_loss = domain_source_loss + domain_target_loss
        return class_loss, domain_loss

class ADDA(nn.Module):
    pass

class WGANDA(nn.Module):
    pass

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
    def __init__(self, input_dim, hidden_dim, num_labels):
        super(REx, self).__init__()
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
