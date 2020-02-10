from model import *
from dataloader import *
from loss import *
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

set_seed()
path = './data/'
patchsize = (64,64)
margin = (80,80)
batch_size = 8
num_workers = 1
epochs = 100
z = 256
h_dim = (16, 32, 64, 256)
input_size = (1,128,128)
lamda = torch.tensor(0.5)
beta = torch.tensor(1.0)
lr = 1e-4

lamda = lamda.to(device)
beta = beta.to(device)
# train_data = cevae(path,patchsize,margin)
# train_loader = DataLoader(train_data,batch_size,num_workers)

train_loader, val_loader = prepare_data(path,margin,patchsize,split = 0.2)
model = VAE(input_size,h_dim,z)

optimizer = optim.Adam(model.parameters(), lr=lr)
lr_scheduler = StepLR(optimizer, step_size=1)

log_path = './logs/'
writer = SummaryWriter(f'{log_path}cevae_{patchsize[0]}_{batch_size}_{lr}')
model.to(device)
model.train()
epoch_train_loss = 0
epoch_val_loss = 0
for epoch in range(epochs):
    train_loss = 0
    for batch_idx, data in enumerate(tqdm(train_loader,desc = 'train_iter',leave=False)):
        inpt, inpt_noisy = data
        
        inpt = inpt.to(device)
        inpt_noisy = inpt_noisy.to(device)
        optimizer.zero_grad()
        x_rec_vae, z_dist,std = model(inpt)
        x_rec_ce,_,_ = model(inpt_noisy)

        # kl_loss_low = kl_loss(z_dist_low)
        kl_loss,kl_div,joint_nll = kl_loss_fn(x_rec_vae,inpt,z_dist,std)
        rec_loss_vae = rec_loss_fn(x_rec_vae, inpt)
        # pdb.set_trace()
        loss_vae = kl_loss * beta + rec_loss_vae

        rec_loss_ce = rec_loss_fn(x_rec_ce, inpt)

        loss_ce = rec_loss_ce

        loss = (1. - lamda) * loss_vae + lamda * loss_ce

        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        writer.add_scalar('ItrLoss/train',loss.item(),epoch*len(train_loader)+batch_idx)
    epoch_train_loss = train_loss/len(train_loader)

    model.eval()
    val_loss = 0
    for idx, vdata in enumerate(tqdm(val_loader,desc = 'val_iter',leave = False)):
        val_inpt, val_noisy = vdata

        val_inpt, val_noisy = val_inpt.to(device), val_noisy.to(device)
        v_rec_vae, v_z, vstd = model(val_inpt)
        v_rec_ce,_,_ = model(val_noisy)
        kl_loss_val = kl_loss_fn(v_rec_vae,val_inpt,v_z,vstd)
        rec_loss_vae_val = rec_loss_fn(v_rec_vae,val_inpt)
        v_loss_vae = kl_loss_val* beta + rec_loss_vae_val

        rec_loss_ce_val = rec_loss_fn(v_rec_ce,val_inpt)
        v_loss_ce = rec_loss_ce_val

        v_loss = (1. - lamda)* v_loss_vae + lamda * v_loss_ce

        val_loss += v_loss.item()
        writer.add_scalar('ItrLoss/Val',v_loss.item(),epoch*len(val_loader)+idx)
    epoch_val_loss = val_loss/len(val_loader)

    writer.add_scalars('EpsLoss/',{'train':epoch_train_loss,'val':epoch_val_loss},epoch) 

