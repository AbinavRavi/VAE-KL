import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader,Dataset
import skimage.transform as skt
from glob import glob
from old_model import VAE
from image_transforms import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

set_seed(616)

def normalise_mask(array):
    narray = array[:] - np.min(array) / (np.max(array) - np.min(array))
    return narray

def save_masks(array,filename,path):
    np.save(path+filename+'_mask',array)

class load_data(Dataset):
    def __init__(self,path,resize,transforms = None,mask=False):
        self.path = path
        self.dataset = sorted(glob(path+'*.nii.gz',recursive=True))
        self.transforms = transforms
        self.mask = False
        self.resize = resize

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        file = self.dataset[index]
        filename = file.split('/')[-1]
        filename = filename.split('.')[0]
        image = nib.load(file)
        x = image.get_data()
        x = skt.resize(x,self.resize)
        x = to_tensor(x).float()
        
        return x, filename

##Parameters
abnormal_path = './all_abnormal/'
save_path = './masks/VAE_masks/'
resize = (128,128)
batch_size = 1
num_workers = 1
model_path = './VAE_models/VAE_V2_1024_0.0001_684.pt'
z_dim = 256
h_dim = (16, 32, 64, 256)
input_size = (1,128,128)

data = load_data(abnormal_path,resize)
dataloader = DataLoader(data,batch_size=batch_size,num_workers=num_workers)

model_module = VAE(input_size,h_dim,z_dim)
model = torch.load(model_path,map_location=device)
model = model.to(device)

model.eval()
with torch.no_grad():
    for idx,(data,filename) in enumerate(dataloader):
        data = data.float()
        data = data.to(device)
        output,mu,std = model(data)
        mask = output - data
        mask = mask.detach().cpu().numpy()
        mask = np.squeeze(mask,axis=0)
        mask = normalise_mask(mask[0,:,:])
        mask = mask < 0.4
        save_masks(mask,filename[0],save_path)