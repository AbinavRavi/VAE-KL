from old_model import *
from dataloader import *
from loss import *
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
<<<<<<< HEAD
from torchvision.utils import make_grid
=======
>>>>>>> b6b4bcfcdced2a597e62f7049b162ce01f49a97b
from tqdm import tqdm
import pdb
import copy
from datetime import date
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

<<<<<<< HEAD
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
=======
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
>>>>>>> b6b4bcfcdced2a597e62f7049b162ce01f49a97b

def set_seed(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

set_seed(123)
path = './data/'
patchsize = (64,64)
margin = (80,80)
batch_size = 1024
num_workers = 2
epochs = 1000
z = 512
h_dim = (16, 32, 64, 256,512)
num_workers = 1
epochs = 1000
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

train_loader, val_loader = vae_data(path,batch_size = batch_size,split = 0.2)
model = VAE(input_size,h_dim,z)
# model = torch.load('models/VAE_1024_0.0001_197.pt')

optimizer = optim.Adam(model.parameters(), lr=lr)
lr_scheduler = StepLR(optimizer, step_size=1)

def weights_init(m):
    classname = m.__class__.__name__
    if type(m) == nn.Conv2d:
        nn.init.normal_(m.weight.data, 0.0, 0.001)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.002)
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

def image_writer(writer,tag,images,epoch):
    """
    write the image into a grid in tensorboard
    tag: the tag for images
    images: can be a tensor, numpy array or PIL object
    epoch: epoch number
    """
    img_grid = make_grid(images,nrow=8,normalize=False)
    writer.add_image(tag,img_grid,epoch)

log_path = './working_logs/'
writer = SummaryWriter(f'{log_path}{date.today()}_vae_{patchsize[0]}_{batch_size}_{lr}')
model.to(device)
# model.apply(weights_init)
model.train()
epoch_train_loss = []
epoch_val_loss = []
for i in range(epochs):
    train_loss = []
    for batch_idx, data in enumerate(tqdm(train_loader,desc = 'train_iter',leave=False)):
        # inpt, inpt_noisy = data
        inpt = data
        
        inpt = inpt.to(device)
        # inpt_noisy = inpt_noisy.to(device)
        optimizer.zero_grad()
        x_rec_vae, z_dist,std = model(inpt)
        # x_rec_ce,_,_ = model(inpt_noisy)

        # kl_loss_low = kl_loss(z_dist_low)
        # kl_loss,kl_div,joint_nll = kl_loss_fn(x_rec_vae,inpt,z_dist,std)
        loss = kl_loss_fn_train(x_rec_vae,inpt,z_dist,std)
        # rec_loss_vae = rec_loss_fn(x_rec_vae, inpt)
        # pdb.set_trace()

        # loss_vae = rec_loss_vae  # kl_loss * beta  

        # rec_loss_ce = rec_loss_fn(x_rec_ce, inpt)

        # loss_ce = rec_loss_ce

        # loss = (1. - lamda) * loss_vae + lamda * loss_ce
        # loss = kl_loss
        loss.backward()
        train_loss.append(loss.item())
        optimizer.step()
        writer.add_scalar('ItrLoss/train',loss.item(),i*len(train_loader)+batch_idx)
    epoch_train_loss = np.array(train_loss).mean()

    model.eval()
    val_loss = []
    for idx, vdata in enumerate(tqdm(val_loader,desc = 'val_iter',leave = False)):
        # val_inpt, val_noisy = vdata
        val_inpt = vdata
        val_inpt = val_inpt.to(device)

        # val_inpt, val_noisy = val_inpt.to(device), val_noisy.to(device)
        v_rec_vae, v_z, vstd = model(val_inpt)
        # v_rec_ce,_,_ = model(val_noisy)
        # kl_loss_val,v_kl,v_joint = kl_loss_fn(v_rec_vae,val_inpt,v_z,vstd)
        v_loss = kl_loss_fn_train(v_rec_vae,val_inpt,v_z,vstd)
        image = v_rec_vae.detach().cpu()
        # rec_loss_vae_val = rec_loss_fn(v_rec_vae,val_inpt)
        # v_loss_vae =  rec_loss_vae_val # kl_loss_val* beta

        # rec_loss_ce_val = rec_loss_fn(v_rec_ce,val_inpt)
        # v_loss_ce = rec_loss_ce_val

        # v_loss = (1. - lamda)* v_loss_vae + lamda * v_loss_ce
        # v_loss = kl_loss_val
        val_loss.append(v_loss.item())
        writer.add_scalar('ItrLoss/Val',v_loss.item(),i*len(val_loader)+idx)
    image_writer(writer,'Itr/Reconstruction',image,i)
    epoch_val_loss = np.array(val_loss).mean()
    # plot_grad_flow(copy.deepcopy(model))
    writer.add_scalars('EpochLoss/',{'train':epoch_train_loss,'val':epoch_val_loss},i)
    print('epoch:{} \t'.format(i+1),'trainloss:{}'.format(epoch_train_loss),'\t','valloss:{}'.format(epoch_val_loss)) 
    if((i+1) % 4 == 0): 
        torch.save(model,'./VAE_models/VAE_V2_{}_{}_{}.pt'.format(batch_size,lr,i+1))


