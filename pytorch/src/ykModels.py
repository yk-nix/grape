# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 13:34:25 2021

@author: yoka
"""
import torch
import torch.nn.functional as F


class FullConnectedNetModel(torch.nn.Module):
    def __init__(self, in_featuers, out_features):
        super(FullConnectedNetModel, self).__init__()
        self.in_features = in_featuers
        self.linear1 = torch.nn.Linear(in_featuers, 512)
        self.linear2 = torch.nn.Linear(512, 256)
        self.linear3 = torch.nn.Linear(256, 128)
        self.linear4 = torch.nn.Linear(128, 64)
        self.linear5 = torch.nn.Linear(64, out_features)
    
    def forward(self, x):
        x = x.view(-1, self.in_features)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        return self.linear5(x)




class ConvolutionalNetModel(torch.nn.Module):
    def __init__(self, in_channels, width, height, out_features):
        assert (width % 4 == 0 and height % 4 == 0), 'the width, height must be the number of times of 4.'
        super(ConvolutionalNetModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=in_channels,
                                     out_channels=10,
                                     kernel_size=5,
                                     padding=2)
        self.conv2 = torch.nn.Conv2d(in_channels=10,
                                     out_channels=20,
                                     kernel_size=5,
                                     padding=2)
        self.maxPooling = torch.nn.MaxPool2d(2)
        width = int(width / 4)
        height = int(height / 4)
        self.fc = FullConnectedNetModel(20 * width * height, out_features)
        #self.linear = torch.nn.Linear(20 * width * height, out_features)
        
    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.maxPooling(self.conv1(x)))
        x = F.relu(self.maxPooling(self.conv2(x)))
        x = x.view(batch_size, -1)
        return self.fc(x)


    

class Inception(torch.nn.Module):
    def __init__(self, in_channels):
        super(Inception, self).__init__()
        self.branch_1x1_conv = torch.nn.Conv2d(in_channels,
                                              out_channels=16,
                                              kernel_size=1)
        self.branch_avg_conv = torch.nn.Conv2d(in_channels,
                                               out_channels=24,
                                               kernel_size=1)
        self.branch_5x5_conv1 = torch.nn.Conv2d(in_channels,
                                               out_channels=16,
                                               kernel_size=1)
        self.branch_5x5_conv2 = torch.nn.Conv2d(in_channels=16,
                                               out_channels=24,
                                               kernel_size=5,
                                               padding=2)
        self.branch_3x4_conv1 = torch.nn.Conv2d(in_channels,
                                                out_channels=16,
                                                kernel_size=1)
        self.branch_3x4_conv2 = torch.nn.Conv2d(in_channels=16,
                                                out_channels=24,
                                                kernel_size=3,
                                                padding=1)
        self.branch_3x4_conv3 = torch.nn.Conv2d(in_channels=24,
                                                out_channels=24,
                                                kernel_size=3,
                                                padding=1)
    def forward(self, x):
        branch_avg = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_avg = self.branch_avg_conv(branch_avg)
        branch_1x1 = self.branch_1x1_conv(x)
        branch_5x5 = self.branch_5x5_conv1(x)
        branch_5x5 = self.branch_5x5_conv2(branch_5x5)
        branch_3x3 = self.branch_3x4_conv1(x)
        branch_3x3 = self.branch_3x4_conv2(branch_3x3)
        branch_3x3 = self.branch_3x4_conv3(branch_3x3)
        outputs = [branch_1x1, branch_5x5, branch_3x3, branch_avg]
        return torch.cat(outputs, dim=1)



class InceptionConvolutionalNetModel(torch.nn.Module):
    def __init__(self, in_channels, width, height, out_features):
        assert (width % 4 == 0 and height % 4 == 0), 'the width, height must be the number of times of 4.'
        super(InceptionConvolutionalNetModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=in_channels,
                                     out_channels=10,
                                     kernel_size=5,
                                     padding=2)
        self.conv2 = torch.nn.Conv2d(in_channels=88,
                                     out_channels=20,
                                     kernel_size=5,
                                     padding=2)
        self.incept1 = Inception(10)
        self.incept2 = Inception(20)
        self.maxPooling = torch.nn.MaxPool2d(2)
        width = int(width / 4)
        height = int(height / 4)
        #self.linear = torch.nn.Linear(88 * width * height, out_features)
        self.fc = FullConnectedNetModel(88 * width * height, out_features)
        
    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.maxPooling(self.conv1(x)))
        x = self.incept1(x)
        x = F.relu(self.maxPooling(self.conv2(x)))
        x = self.incept2(x)
        x = x.view(batch_size, -1)
        return self.fc(x)



class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock,self).__init__()
        self.channels = channels
        self.conv1 = torch.nn.Conv2d(channels, channels,
                                     kernel_size=3, 
                                     padding=1)
        self.conv2 = torch.nn.Conv2d(channels, channels,
                                     kernel_size=3,
                                     padding=1)
    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        return F.relu(x+y)




class ResidualConvolutionalNetModel(torch.nn.Module):
    def __init__(self, in_channels, width, height, out_features):
        assert (width % 4 == 0 and height % 4 == 0), 'the width, height must be the number of times of 4.'
        super(ResidualConvolutionalNetModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=in_channels,
                                     out_channels=16,
                                     kernel_size=5,
                                     padding=2)
        self.conv2 = torch.nn.Conv2d(in_channels=16,
                                     out_channels=32,
                                     kernel_size=5,
                                     padding=2)
        self.rblock1 = ResidualBlock(16)
        self.rblock2 = ResidualBlock(32)
        self.maxPooling = torch.nn.MaxPool2d(2)
        width = int(width / 4)
        height = int(height / 4)
        #.linear = torch.nn.Linear(64 * width * height, out_features)
        self.fc = FullConnectedNetModel(32 * width * height, out_features)
        
    def forward(self, x):
        batch_size = x.size(0)
        x = self.maxPooling(F.relu(self.conv1(x)))
        x = self.rblock1(x)
        x = self.maxPooling(F.relu(self.conv2(x)))
        x = self.rblock2(x)        
        x = x.view(batch_size, -1)
        return self.fc(x)
  
   
      