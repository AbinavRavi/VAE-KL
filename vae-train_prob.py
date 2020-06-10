from old_model import *
from dataloader import *
from loss import *
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm
import pdb
import copy
from datetime import date
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def set_seed(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

set_seed(123)
path = './data/'

batch_size = 1024
num_workers = 2
epochs = 1000
z = 256
h_dim = (16, 32, 64, 256)
input_size = (1,128,128)
lamda = torch.tensor(0.5)
beta = torch.tensor(1.0)
lr = 1e-4

lamda = lamda.to(device)
beta = beta.to(device)


train_loader, val_loader = vae_data(path,batch_size = batch_size,split = 0.2)
model = VAE(input_size,h_dim,z)

optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(optimizer, mode='min',patience=3)

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
writer = SummaryWriter(f'{log_path}{date.today()}_vae_V2_{batch_size}_{lr}')
model.to(device)
model.apply(weights_init)
model.train()
epoch_train_loss = []
epoch_val_loss = []
for i in range(epochs):
    train_loss = []
    for batch_idx, data in enumerate(tqdm(train_loader,desc = 'train_iter',leave=False)):
        inpt = data
        
        inpt = inpt.to(device)
        optimizer.zero_grad()
        x_rec_vae, z_dist,std = model(inpt)
        # loss = kl_loss_fn_train(x_rec_vae,inpt,z_dist,std)
        loss, kl, rec = loss_function(x_rec_vae,inpt,z_dist,std)
        loss.backward()
        train_loss.append(loss.item())
        optimizer.step()
        writer.add_scalar('ItrLoss/train',loss.item(),i*len(train_loader)+batch_idx)
    epoch_train_loss = np.array(train_loss).mean()
    scheduler.step(epoch_train_loss)
    model.eval()
    val_loss = []
    for idx, vdata in enumerate(tqdm(val_loader,desc = 'val_iter',leave = False)):
        # val_inpt, val_noisy = vdata
        val_inpt = vdata
        val_inpt = val_inpt.to(device)

        
        v_rec_vae, v_z, vstd = model(val_inpt)
        
        v_loss,v_kl,v_joint = loss_function(v_rec_vae,val_inpt,v_z,vstd)
        # v_loss = kl_loss_fn_train(v_rec_vae,val_inpt,v_z,vstd)
        image = v_rec_vae.detach().cpu()
        val_loss.append(v_loss.item())
        writer.add_scalar('ItrLoss/Val',v_loss.item(),i*len(val_loader)+idx)
    image_writer(writer,'Itr/Reconstruction',image,i)
    epoch_val_loss = np.array(val_loss).mean()
    # plot_grad_flow(copy.deepcopy(model))
    writer.add_scalars('EpochLoss/',{'train':epoch_train_loss,'val':epoch_val_loss},i)
    print('epoch:{} \t'.format(i+1),'trainloss:{}'.format(epoch_train_loss),'\t','valloss:{}'.format(epoch_val_loss)) 
    if((i+1) % 4 == 0): 
        torch.save(model,'./VAE_models/VAE_V2_{}_{}_{}.pt'.format(batch_size,lr,i+1))


