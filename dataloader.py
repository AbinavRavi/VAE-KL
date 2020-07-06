import numpy as np
import nibabel as nib
import torchvision.transforms as transforms
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable as V
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader 
from PIL import Image
from image_transforms import *
import glob
import pdb

class cevae(Dataset):
    def __init__(self,path,patchsize,margin,transforms = None,mask=False):
        self.path = path
        self.dataset = glob.glob(path+'*.nii.gz',recursive=True)
        self.patchsize = patchsize
        self.transforms = transforms
        self.margin = margin
        self.mask = False

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image = nib.load(self.dataset[index])
        x = image.get_data()
        
        mask = square_mask(x,self.margin,self.patchsize)
#         if(self.mask == True):
#             x = square_mask(x,self.margin,self.patchsize)
        # x = RandomCrop(x,192)
        # x = np.expand_dims(x,axis=2)
        x = RandomHorizontalFlip(x)
        x = Resize(x,(128,128))
        x = normalise(x)
        x = np.expand_dims(x,axis=2)
        x = to_tensor(x).float()
        masked_image = torch.where(mask !=0,mask,x)
#         if self.transforms:
#             x = self.transforms(x)
        return x #,masked_image
            
class vae_loader(Dataset):
    def __init__(self,path,transforms = None,mask=False):
        self.path = path
        self.dataset = glob.glob(path+'*.nii.gz',recursive=True)
        self.transforms = transforms
        self.mask = False

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image = nib.load(self.dataset[index])
        x = image.get_data()
        x = RandomHorizontalFlip(x)
        x = Resize(x,(128,128))
        x = normalise(x)
        x = np.expand_dims(x,axis=2)
        x = to_tensor(x).float()
        
        return x

        
def prepare_data(path,margin,patchsize,batch_size =8,split = 0.2):
    dataset = cevae(path,margin,patchsize)
    val_size = int(split*len(dataset))
    train_size = int((1-split)*len(dataset))
    train_ds, val_ds = random_split(dataset,[train_size,val_size])
    return DataLoader(train_ds, batch_size=batch_size,num_workers=1), DataLoader(val_ds, batch_size=batch_size, num_workers=1)
    


def vae_data(path,batch_size=8,split=0.2):
    dataset = vae_loader(path)
    val_size = int(split*len(dataset))
    train_size = int((1-split)*len(dataset))
    train_ds, val_ds = random_split(dataset,[train_size,val_size])
    return DataLoader(train_ds, batch_size=batch_size,num_workers=1), DataLoader(val_ds, batch_size=batch_size, num_workers=1)        
