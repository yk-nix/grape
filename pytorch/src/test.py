# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#from PIL import Image
import torch
import numpy as np
import math
import matplotlib.pyplot as plt
from ykDataloader import MnistDataLoader
from ykDataloader import Cifa10DataLoader
from ykModels import FullConnectedNetModel
from ykModels import ConvolutionalNetModel
    

def showImages(images, start, end):
    num = end - start
    nrows = ncols = int(math.sqrt(num))
    if nrows * ncols < num :
        ncols += 1
        if nrows * ncols < num:
            nrows += 1
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, 
                sharex=True, sharey=True)
    labels = []    
    if num > 1:
        ax = ax.flatten()
        for i in range(nrows * ncols):
            ax[i].spines['left'].set_visible(False)
            ax[i].spines['right'].set_visible(False)
            ax[i].spines['top'].set_visible(False)
            ax[i].spines['bottom'].set_visible(False)       
        for i in range(start, end):
            labels.append(images[i][1])
            ax[i-start].imshow(images[i][0])
            ax[i-start].set_xticks([])
            ax[i-start].set_yticks([])
    else:
        ax.imshow(images[start][0])
        ax.set_xticks([])
        ax.set_yticks([])
        labels.append(images[start][1])    
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    return labels

def showImage(images, idx):
    return showImages(images, idx, idx+1)

def doFullConnectedNetModelTest(idx):
    mnist = MnistDataLoader(64)
    img = np.array(mnist.testImages[idx][0])
    model = FullConnectedNetModel(28*28, 10)
    model.load_state_dict(torch.load('./save/mnistClassfier-FCN-epoch-007.pt'))
    outputs = model(mnist.transform(img))
    _, predict = torch.max(outputs.data, dim=1)
    print(predict.item())
    showImage(mnist.testImages, idx) 

def doConvolutionalNetModelTest(idx):
    mnist = MnistDataLoader(64)
    img = np.array(mnist.testImages[idx][0])
    model = ConvolutionalNetModel(1, 10)
    model(mnist.transform(img).unsqueeze(0))
    
if __name__ == '__main__':
    cifa = Cifa10DataLoader(32, download=False)
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    '''
    i = 0
    while True:
        showImage(cifa.trainImages, i) 
        print(classes[cifa.trainImages[i][1]])
        i += 1
        input('')
    '''
        
     
    

