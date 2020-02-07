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
    def __init__(self,path,patchsize,margin,transforms = None):
        self.path = path
        self.dataset = glob.glob(path+'*.nii.gz',recursive=True)
        self.patchsize = patchsize
        self.transforms = transforms
        self.margin = margin

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image = nib.load(self.dataset[index])
        x = image.get_data()
        x = square_mask(x,self.margin,self.patchsize)
        # x = RandomCrop(x,192)
        x = RandomHorizontalFlip(x)
        x = Resize(x,(128,128))
        x = np.expand_dims(x,axis=2)
        x = to_tensor(x)
        
        if self.transforms:
            x = self.transforms(x)

        return x
            

# def get_data(path,split,mode='train'):
    


        
