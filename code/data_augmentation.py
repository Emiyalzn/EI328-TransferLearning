import torch
import os
from torch.utils.data import DataLoader
import torch.autograd as autograd
import torch.nn.functional as F
import numpy as np
from parser import parse_arguments
from models import create_model
from dataset import SeedDataset, DATASET_PATH
import scipy.io as sio

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
checkpoint_dir = "../checkpoint"

def calc_gradient_penalty(netD, real_data, fake_data):
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1).to(real_data.device)
    alpha = alpha.expand(real_data.size())
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(interpolates.device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def wGAN_augmentation(args, dataset):
    wGAN = create_model(args)
    wGAN.netD, wGAN.netG = wGAN.netD.to(device), wGAN.netG.to(device)

    # load classifier
    args.model = 'ResNet'
    resnet =  create_model(args).to(device)
    resnet_state_dict = torch.load(os.path.join(checkpoint_dir, 'ResNet_checkpoint.pt'))
    resnet.load_state_dict(resnet_state_dict)
    resnet.eval()

    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)
    data_iter = iter(dataloader)

    optimizerD = torch.optim.Adam(wGAN.netD.parameters(), lr=args.lr, betas=(0.5, 0.9))
    optimizerG = torch.optim.Adam(wGAN.netG.parameters(), lr=args.lr, betas=(0.5, 0.9))

    one = torch.tensor(1, dtype=torch.float)
    mone = one * -1

    flag = 0
    auxiliary_X = None
    auxiliary_Y = None

    for iteration in range(args.gen_iters):
        for p in wGAN.netD.parameters():
            p.requires_grad = True

        # update D network
        for _ in range(args.critic_iters):
            try:
                real_data, _ = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                real_data, _ = next(data_iter)
            real_data_v = autograd.Variable(real_data).to(device, dtype=torch.float)

            wGAN.netD.zero_grad()

            # train with real
            D_real = wGAN.netD(real_data_v).mean()
            D_real.backward(mone)

            # train with fake
            noise = torch.randn(args.batch_size, 64).to(device)
            noisev = autograd.Variable(noise, volatile=True)
            fake = autograd.Variable(wGAN.netG(noisev).data)
            inputv = fake
            D_fake = wGAN.netD(inputv).mean()
            D_fake.backward(one)

            # train with gradient penalty
            gradient_penalty = calc_gradient_penalty(wGAN.netD, real_data_v.data, fake.data) * args.wgan_lamda
            gradient_penalty.backward()

            D_cost = D_fake - D_real + gradient_penalty
            # Wasserstein_D  = D_real - D_fake
            optimizerD.step()

        for p in wGAN.netD.parameters():
            p.requires_grad = False
        wGAN.netG.zero_grad()

        noise = torch.randn(args.batch_size, 64).to(device)
        noisev = autograd.Variable(noise)
        fake = wGAN.netG(noisev)
        G = wGAN.netD(fake).mean()
        G.backward(mone)
        G_cost = -G
        optimizerG.step()

        # measure confidence
        class_logits = resnet(fake)
        class_prob, class_label = torch.max(F.softmax(class_logits, dim=1), dim=1)

        # record those generated data with high confidence
        generate_X = fake[class_prob > .9].detach().cpu().numpy()
        generate_Y = class_label[class_prob > .9].detach().cpu().numpy() - 1

        if generate_Y.shape[0] > 0:
            if flag == 0:
                flag = 1
                auxiliary_X = generate_X
                auxiliary_Y = generate_Y
            else:
                auxiliary_X = np.concatenate((auxiliary_X, generate_X), axis=0)
                auxiliary_Y = np.concatenate((auxiliary_Y, generate_Y), axis=0)
            # early stop with enough data
            if auxiliary_Y.shape[0] > 1000:
                break

        print(f"Iteration {iteration}, D loss: {D_cost:.4f}, G loss: {G_cost:.4f}")

    sio.savemat(os.path.join(DATASET_PATH, f"auxiliary_data_label_{auxiliary_Y[1000]}.mat"), {'X': auxiliary_X, 'Y': auxiliary_Y})

if __name__ == '__main__':
    dataset = SeedDataset(True)
    dataset.prepare_gen()

    args = parse_arguments()
    wGAN_augmentation(args, dataset)

