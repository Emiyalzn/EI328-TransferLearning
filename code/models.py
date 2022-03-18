from torch.autograd import Function
import torch.nn as nn

def create_model(args):
    if args.model == 'DANN':
        return DANN(310, args.hidden_dim, 3, 2, args.lamda)
    elif args.model == 'MLP':
        return MLP(310, args.hidden_dim, 3)

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

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_labels):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels

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