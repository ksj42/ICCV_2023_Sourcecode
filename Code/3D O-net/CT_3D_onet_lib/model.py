import torch
import torch.nn as nn

from .config import *

def split_front_and_rear(x):

    x_front = x[:,:,0:BOUNDARY,:,:]
    x_rear = x[:,:,BOUNDARY:96,:,:]

    return x_front, x_rear

class UNet_3D(nn.Module):
    def __init__(self):
        super(UNet_3D, self).__init__()

        # Convolution + BatchNormalization + Relu 
        def CBR3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True): 
            layers = []
            layers += [nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)]
            layers += [nn.BatchNorm3d(num_features=out_channels)]
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers)

            return cbr


        # Contracting path
        self.enc1_1 = CBR3d(in_channels=1, out_channels=64)
        self.enc1_2 = CBR3d(in_channels=64, out_channels=64)

        self.pool1 = nn.MaxPool3d(kernel_size=2)

        self.enc2_1 = CBR3d(in_channels=64, out_channels=128)
        self.enc2_2 = CBR3d(in_channels=128, out_channels=128)

        self.pool2 = nn.MaxPool3d(kernel_size=2)

        self.enc3_1 = CBR3d(in_channels=128, out_channels=256)
        self.enc3_2 = CBR3d(in_channels=256, out_channels=256)

        self.pool3 = nn.MaxPool3d(kernel_size=2)

        self.enc4_1 = CBR3d(in_channels=256, out_channels=512)

        # Expansive path
        self.dec4_1 = CBR3d(in_channels=512, out_channels=256)

        self.unpool3 = nn.ConvTranspose3d(in_channels=256, out_channels=256,
                                        kernel_size=2, stride=2, padding=0, bias=True)

        self.dec3_2 = CBR3d(in_channels=2 * 256, out_channels=256)
        self.dec3_1 = CBR3d(in_channels=256, out_channels=128)

        self.unpool2 = nn.ConvTranspose3d(in_channels=128, out_channels=128,
                                        kernel_size=2, stride=2, padding=0, bias=True)

        self.dec2_2 = CBR3d(in_channels=2 * 128, out_channels=128)
        self.dec2_1 = CBR3d(in_channels=128, out_channels=64)

        self.unpool1 = nn.ConvTranspose3d(in_channels=64, out_channels=64,
                                        kernel_size=2, stride=2, padding=0, bias=True)

        self.dec1_2 = CBR3d(in_channels=2 * 64, out_channels=64)
        self.dec1_1 = CBR3d(in_channels=64, out_channels=64)

    def forward(self, x):

        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)
        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        dec4_1 = self.dec4_1(enc4_1)

        unpool3 = self.unpool3(dec4_1) 
        cat3 = torch.cat((unpool3, enc3_2), dim=1)     
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)

        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, enc2_2), dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)

        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, enc1_2), dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        return dec1_1

class ONet_3D(nn.Module):
    def __init__(self, UNet_3D):
        super(ONet_3D, self).__init__()
        self.net_front = UNet_3D()
        self.net_rear = UNet_3D()
        self.fc = nn.Conv3d(in_channels=32*2, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
   
    # forward
    def forward(self, x):
        x_front, x_rear = split_front_and_rear(x)
        y_front = self.net_front(x_front)
        y_rear = self.net_front(x_rear)
        cat = torch.cat((y_front, y_rear), dim=2)
        y = self.fc(cat)
        return y
