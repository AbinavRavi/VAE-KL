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

class cevae(Dataset):
    def __init__(self,path,patchsize,transforms=None):
        self.path = path
        self.dataset = glob.glob(path+'*.nii.gz',recursive=True)
        self.patchsize = patchsize
        self.transforms = transforms

    def __len__(self):
        return self.dataset

    def __getitem__(self, index):
        image = nib.load(self.dataset[index//num_perts])
        x = image.get_data()
        x = torch.from_numpy(x)
        
        if self.transforms:
            x = self.transforms(x)

        return x
            



        
