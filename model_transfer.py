# model.py
import torch
from torch import nn as nn
from torchvision import models

#%% TrackingModel

class TrackingModel(nn.Module):
    def __init__(self,in_channels=1,img_size=(1024,1024),output=2):
        super().__init__()
        # in_channels = 1, img_size = (1024,1024), output = 2
        
        # Import ResNet18
        # from (N,1,1024,1024) to (N,64,512,512) to (N,64,256,256) to (N,128,128,128)
        # to (N,256,64,64) to (N,512,32,32) 
        model = models.resnet18(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
            
        # change adaptive average pooling to average pooling
        # average pool from (N,512,32,32) to (512,16,16)
        model.avgpool = nn.AvgPool2d(2,2)
        model.fc = nn.Identity() # remove fc layer
        
        # define the modeified resnet18 
        self.resnet18_pre = model
        
        # from (N*131,072) to (N*64)
        self.linear1 = nn.Linear(in_channels*512*16*16, in_channels*64)
        
        # bn after first linear
        self.bn = nn.BatchNorm1d(in_channels*64)
        
        # ReLU layer after first linear
        self.relu = nn.ReLU(inplace=True)
        
        # from (N*64) to (N*4)
        self.linear2 = nn.Linear(in_channels*64, 2)
        
        # Sigmoid layer after second linear
        self.sigmoid = nn.Sigmoid()
        
        # initialize weights
        self._init_weights()
        
    def _init_weights(self):
    # initiate with Xavier initialization
        for m in self.modules():
            if type(m) in {nn.Conv2d,nn.Linear}:
                # Weight of layers
                nn.init.xavier_normal_(m.weight)
                # if we have bias
                if m.bias is not None:
                    m.bias.data.fill_(0.01)  
                        
            if type(m) in {nn.BatchNorm2d}:
                # Weight of layers
                nn.init.normal_(m.weight)
                # if we have bias
                if m.bias is not None:
                    m.bias.data.fill_(0.01) 

    def forward(self, X):
        X = self.resnet18_pre(X)
        
        X = X.view(X.size(0),-1) # flatten to 1D vector
        X = self.linear1(X)
        X = self.bn(X)
        X = self.relu(X)
        X = self.linear2(X)
        X = self.sigmoid(X)
        
        return X

#%% Helper function
class ConvBlock(nn.Module):
    
    def __init__(self,input_size,output_size):
        super().__init__()
        # from (N*input_size*H*W) to (N*output_size*2,H/2,W/2)
        
        # define conv + bn + relu
        self.conv1 = nn.Conv2d(input_size,output_size,kernel_size=3,padding=1)
        self.bn1 = nn.BatchNorm2d(output_size)
        self.relu1 = nn.ReLU(inplace=True)
        
        # define conv + bn + relu + maxpool
        self.conv2 = nn.Conv2d(output_size,output_size,kernel_size=3,padding=1)
        self.bn2 = nn.BatchNorm2d(output_size)
        self.relu2 = nn.ReLU(inplace=True)
        self.max = nn.MaxPool2d(2,2)
        
    def forward(self, X):
        X = self.conv1(X)
        X = self.bn1(X)
        X = self.relu1(X)
        
        X = self.conv2(X)
        X = self.bn2(X)
        X = self.relu2(X)
        
        X = self.max(X)
        
        return X

















