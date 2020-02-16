import numpy as np
import torch 
import torch.nn as nn
import torch.distributions as dist

class Encoder(nn.Module):
    def __init__(self,input_size):
        super(Encoder,self).__init__()

        self.conv1 = nn.Conv2d(input_size,16,kernel_size =4,stride=2,padding=1,bias = False)
        self.conv2 = nn.Conv2d(16,64,kernel_size=4,stride=2,padding=1,bias = False)
        self.conv3 = nn.Conv2d(64,128,kernel_size=4,stride=2,padding=1,bias = False)
        self.conv4 = nn.Conv2d(128,256,kernel_size=4,stride=1,bias = False)
        # nn.InstanceNorm2d = nn.InstanceNorm2d()
        self.lrelu = nn.LeakyReLU(inplace=True)

    def forward(self,x):
        x = self.conv1(x)
        # x = nn.BatchNorm2d(nn.InstanceNorm2d(x))
        x = self.lrelu(x)
        x = self.conv2(x)
        # x = nn.BatchNorm2d(nn.InstanceNorm2d(x))
        x = self.lrelu(x)
        x = self.conv3(x)
        # x = nn.BatchNorm2d(nn.InstanceNorm2d(x))
        x = self.lrelu(x)
        x = self.conv4(x)

        return x

class Decoder(nn.Module):
    def __init__(self,z_dim):
        super(Decoder,self).__init__()

        self.deconv1 = nn.ConvTranspose2d(z_dim,128,kernel_size=4,stride=1,bias = False)
        self.deconv2 = nn.ConvTranspose2d(128, 64,kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv3 = nn.ConvTranspose2d(64,1,kernel_size=4, stride=2, padding=1,bias = False)
        # nn.InstanceNorm2d = nn.InstanceNorm2d()
        self.lrelu = nn.LeakyReLU()

    def forward(self,x):
        x = self.deconv1(x)
        # x = nn.BatchNorm2d(nn.InstanceNorm2d(x))
        x = self.lrelu(x)
        x = self.deconv2(x)
        # x = nn.BatchNorm2d(nn.InstanceNorm2d(x))
        x = self.lrelu(x)
        x = self.deconv3(x)

        return x

class VAE(nn.Module):
    def __init__(self,input_size,z_dim):
        super(VAE,self).__init__()

        self.encoder = Encoder(input_size=1)
        self.decoder = Decoder(z_dim=256)

    def forward(self,input):
        x = input

        enc = self.encoder(x)
        print(enc.shape)
        mu, log_std = torch.chunk(enc, 2, dim=1)
        std = torch.exp(log_std)
        z_dist = dist.Normal(mu, std)
        print(z_dist)
        # x_rec = self.decoder(z_dist)

        # return x_rec,mu,std



    