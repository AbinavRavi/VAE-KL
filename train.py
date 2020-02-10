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
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

set_seed(123)
path = './data/'
patchsize = (64,64)
margin = (80,80)
batch_size = 64
num_workers = 1
epochs = 100
z = 256
h_dim = (16, 32, 64, 256)
input_size = (1,128,128)
lamda = torch.tensor(0.5)
beta = torch.tensor(1.0)
lr = 2e-7

lamda = lamda.to(device)
beta = beta.to(device)
# train_data = cevae(path,patchsize,margin)
# train_loader = DataLoader(train_data,batch_size,num_workers)

train_loader, val_loader = prepare_data(path,margin,patchsize,batch_size,split = 0.2)
model = VAE(input_size,h_dim,z)

optimizer = optim.Adam(model.parameters(), lr=lr)
lr_scheduler = StepLR(optimizer, step_size=1)

def weights_init(m):
    classname = m.__class__.__name__
    if type(m) == nn.Conv2d:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def plot_grad_flow(model):
        """
            gradient flow tracking
        """
        for n, p in model.named_parameters():
            if (p.requires_grad) and ("bias" not in n):
                ave_grad = p.grad.abs().mean()
                max_grad = p.grad.abs().max()
                writer.add_scalar('average gradient',n, ave_grad.item())
                writer.add_scalar('maximum gradient',str(n), max_grad.item())



log_path = './logs/'
writer = SummaryWriter(f'{log_path}cevae_{patchsize[0]}_{batch_size}_{lr}')
model.to(device)
model.apply(weights_init)
model.train()
epoch_train_loss = []
epoch_val_loss = []
for i in range(epochs):
    train_loss = []
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
        loss_vae = rec_loss_vae  # kl_loss * beta  

        rec_loss_ce = rec_loss_fn(x_rec_ce, inpt)

        loss_ce = rec_loss_ce

        loss = (1. - lamda) * loss_vae + lamda * loss_ce

        loss.backward()
        train_loss.append(loss.item())
        optimizer.step()
        writer.add_scalar('ItrLoss/train',loss.item(),i*len(train_loader)+batch_idx)
    epoch_train_loss = np.array(train_loss).mean()

    model.eval()
    val_loss = []
    for idx, vdata in enumerate(tqdm(val_loader,desc = 'val_iter',leave = False)):
        val_inpt, val_noisy = vdata

        val_inpt, val_noisy = val_inpt.to(device), val_noisy.to(device)
        v_rec_vae, v_z, vstd = model(val_inpt)
        v_rec_ce,_,_ = model(val_noisy)
        kl_loss_val,v_kl,v_joint = kl_loss_fn(v_rec_vae,val_inpt,v_z,vstd)
        rec_loss_vae_val = rec_loss_fn(v_rec_vae,val_inpt)
        v_loss_vae =  rec_loss_vae_val # kl_loss_val* beta

        rec_loss_ce_val = rec_loss_fn(v_rec_ce,val_inpt)
        v_loss_ce = rec_loss_ce_val

        v_loss = (1. - lamda)* v_loss_vae + lamda * v_loss_ce

        val_loss.append(v_loss.item())
        writer.add_scalar('ItrLoss/Val',v_loss.item(),i*len(val_loader)+idx)
    epoch_val_loss = np.array(val_loss).mean()
    # plot_grad_flow(copy.deepcopy(model))
    writer.add_scalars('EpochLoss/',{'train':epoch_train_loss,'val':epoch_val_loss},i)
    print('epoch:{} \t'.format(i+1),'trainloss:{}'.format(epoch_train_loss),'\t','valloss:{}'.format(epoch_val_loss))  
    torch.save(model,'./models/ceVAE_{}_{}_{}.pt'.format(batch_size,lr,i+1))


