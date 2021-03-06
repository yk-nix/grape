# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 10:01:55 2021

@author: yoka
"""

from torchvision.transforms import transforms
from torchvision import datasets
from torch.utils.data import DataLoader


class ykDataLoader:
    def __init__(self, name, rootpath, batchSize, getDatasetFunc, datasetName, transform, download):
        self.name = name
        self.batchSize = batchSize
        self.trainImages = getDatasetFunc(root=rootpath, 
                                          train=True,
                                          download=download)
        self.testImages = getDatasetFunc(root=rootpath,
                                         train=False,
                                         download=download)
        self.trainDataset = getDatasetFunc(root=rootpath,
                                           train=True,
                                           download=download,
                                           transform=transform)
        self.testDataset = getDatasetFunc(root=rootpath,
                                          train=False,
                                          download=download,
                                          transform=transform)
        self.trainDataloader = DataLoader(self.trainDataset,
                                          batch_size=batchSize,
                                          shuffle=True)
        self.testDataloader = DataLoader(self.testDataset,
                                         batch_size=batchSize,
                                         shuffle=False)
class MnistDataLoader(ykDataLoader):
    def __init__(self, batchSize, rootpath, download=True):
        self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
                ])
        super(MnistDataLoader, self).__init__('MNIST',
                                              rootpath,
                                              batchSize, 
                                              datasets.MNIST,
                                              'MNIST', 
                                              self.transform,
                                              download)

class Cifa10DataLoader(ykDataLoader):
    def __init__(self, batchSize, rootpath, download=True):
        self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
        super(Cifa10DataLoader, self).__init__('CIFA10',
                                               rootpath,
                                               batchSize, 
                                               datasets.CIFAR10,
                                               'CIFA10', 
                                               self.transform,
                                               download)
    
    