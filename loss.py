import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import math

def kl_loss_fn(recon_x,x,mu,logstd,rec_log_std=0,sum_samplewise=True):
    """
    recon_x : reconstructed sample  
    x : sample
    mu: mean of sample
    logstd: log of std deviation
    rec_log_std: reconstruction log standard deviation
    Gives the loss function for CeVAE network"""

    rec_std = math.exp(rec_log_std)
    rec_var = rec_std**2

    x_dist = dist.Normal(recon_x,rec_std)
    log_p_x_z = x_dist.log_prob(x)
    if sum_samplewise:
        log_p_x_z = torch.sum(log_p_x_z, dim=(1, 2, 3))

    z_prior = dist.Normal(0, 1.)
    z_post = dist.Normal(mu, torch.exp(logstd))

    kl_div = dist.kl_divergence(z_post, z_prior)
    if sum_samplewise:
        kl_div = torch.sum(kl_div, dim=(1, 2, 3))

    if sum_samplewise:
        loss = torch.mean(kl_div - log_p_x_z)
    else:
        loss = torch.mean(torch.sum(kl_div, dim=(1, 2, 3)) - torch.sum(log_p_x_z, dim=(1, 2, 3)))

    return loss, kl_div, -log_p_x_z

def rec_loss_ce(recon_x, x):

    """
    The function checks the l2 loss for reconstruction of image
    """

    loss_fn = nn.MSELoss()
    loss = loss_fn(x,recon_x)

    return loss

