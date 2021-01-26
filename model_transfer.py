# model.py
import torch
from torch import nn as nn
from torchvision import models

#%% TrackingModel

class TrackingModel(nn.Module):
    def __init__(self):
        super().__init__()

        # define pre trained ResNet, from (N,1,1024,1024) to (N,2048,8,8) 
        self.pre_trained = ResNet(101)
        
        # from (N*131,072) to (N*64) + Batch Normalization + ReLU
        self.linear1 = nn.Linear(2048*8*8,64)
        self.bn = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # from (N*64) to (N*4) + Sigmoid
        self.linear2 = nn.Linear(64, 2)
        self.sigmoid = nn.Sigmoid()
        
        # initialize weights
        self._init_weights()
        
    def _init_weights(self):
    # initiate with Xavier initialization
        for m in self.modules():
            if m != self.pre_trained:
                if type(m) in {nn.Conv2d,nn.Linear}:
                    nn.init.xavier_normal_(m.weight) # Weight of layers
                    if m.bias is not None: # if we have bias
                        m.bias.data.fill_(0.01)  
                                
                if type(m) in {nn.BatchNorm2d}:
                    nn.init.normal_(m.weight) # Weight of layers
                    if m.bias is not None: # if we have bias
                        m.bias.data.fill_(0.01) 

    def forward(self, X):
        X = self.pre_trained(X)
        X = X.view(X.size(0),-1) # flatten to 1D vector
        X = self.linear1(X)
        X = self.bn(X)
        X = self.relu(X)
        X = self.linear2(X)
        X = self.sigmoid(X)
        
        return X


#%% Class ResNet Pretrained

class ResNet(nn.Module):
    def __init__(self,num_of_layers):
        super().__init__()
        # from (N,1,1024,1024) to (N,64,512,512) to (N,64,256,256) to (N,128,128,128)
        # to (N,256,64,64) to (N,512,32,32) 
        
        # choose resnet type:
        if num_of_layers == 18:
            # Import ResNet18 with trained weights
            model = models.resnet18(pretrained=True)
        elif num_of_layers == 34:
             # Import ResNet34 with trained weights
             model = models.resnet34(pretrained=True)
        elif num_of_layers == 50:
             # Import ResNet34 with trained weights
             model = models.resnet50(pretrained=True)
        elif num_of_layers == 101:
             # Import ResNet34 with trained weights
             model = models.resnet101(pretrained=True)
        
        # delete last two layers 
        model.avgpool = nn.MaxPool2d(4,4)
        model.fc = nn.Identity()
        
        # freeze wieghts 
        for param in model.parameters():
            param.requires_grad = False
            
        # define the modeified resnet
        self.resnet_pre = model 
                
    def forward(self,X):
        X = self.resnet_pre(X)
        return X



#%% convblock function
class ConvBlock(nn.Module):
    
    def __init__(self,input_size,output_size,num_of_blocks):
        super().__init__()
        # from (N*input_size*H*W) to (N*output_size,H/2,W/2)
        self.num_of_blocks = num_of_blocks
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_size,output_size,kernel_size=3,padding=1),
            nn.BatchNorm2d(output_size),
            nn.ReLU(inplace=True))
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(input_size,output_size,kernel_size=3,padding=1),
            nn.BatchNorm2d(output_size),
            nn.ReLU(inplace=True))
            
        self.max = nn.MaxPool2d(2,2)
        
    def forward(self, X):
        X = self.conv1(X)
        if self.num_of_blocks == 2: X = self.conv2(X)
        X = self.max(X)
        
        return X
    
