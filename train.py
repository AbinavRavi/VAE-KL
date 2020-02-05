import torch
import torch.optim as optim
from loss import *
import torch.utils.tensorboard import SummaryWriter

def train(epoch, model, optimizer, train_loader, device, tx, log_var_std, lamda=0.5, beta=1.0):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        inpt = data["data"][0].float()

        ce_tensor = data["mask"][0].float()
        inpt_noisy = torch.where(ce_tensor != 0, ce_tensor, inpt)

        inpt = inpt.to(device)
        inpt_noisy = inpt_noisy.to(device)

        optimizer.zero_grad()
        x_rec_vae, z_dist, = model(inpt)
        x_rec_ce = model(inpt_noisy)

        # kl_loss_low = kl_loss(z_dist_low)
        kl_loss = kl_loss_fn(z_dist)
        rec_loss_vae = rec_loss_fn(x_rec_vae, inpt)
        loss_vae = kl_loss * beta + rec_loss_vae

        rec_loss_ce = rec_loss_fn(x_rec_ce, inpt)

        loss_ce = rec_loss_ce

        loss = (1. - lamda) * loss_vae + lamda * loss_ce

        loss.backward()
        train_loss += loss.item()
        optimizer.step()
