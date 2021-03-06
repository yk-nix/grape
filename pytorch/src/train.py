# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 14:42:13 2021

@author: yoka
"""
import torch
import numpy as np
import os
import matplotlib.pyplot as plt

def doTraining(epoch, model, criterion, optimizer, dataloader, printStride, modleName, saveDir):
    runningLoss = 0.0
    for i, (inputs, labels) in enumerate(dataloader, 0):
        optimizer.zero_grad()
        # 1.Forward
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        # 2.Backward
        loss.backward()
        # 3.Update
        optimizer.step()
        # Save training result
        dirpath = saveDir + '/' + modleName
        if not os.path.exists(dirpath) :
            os.mkdir(dirpath)
        torch.save(model.state_dict(),  dirpath + '/' + 'epoch-' + '{:03d}'.format(epoch) + '.pt')
        # Print traing process
        runningLoss += loss.item()
        if i % printStride == printStride - 1:
            print('[%d, %5d]  loss: %.4f'%(epoch + 1, i + 1, runningLoss/printStride))
            runningLoss = 0.0
  
def doTesting(model, dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracyRate = 100 * correct / total
    print('Accuracy rate on test-dataset: %.2f %% (%d / %d)'%(accuracyRate, correct, total))
    return accuracyRate

def imgClassfierTraining(dataset, model, model_name, epoch, 
                         savepath='../save', 
                         resultpath='../results',
                         printStride=100):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.8)
    rates = []
    epochs = []
    for i in range(epoch):
        print('--------------------------')
        print('[epcho, batchIdx] loss-value')
        doTraining(i, model, criterion, optimizer,
                   dataset.trainDataloader, printStride, model_name, savepath)
        rate = doTesting(model, dataset.testDataloader)
        epochs.append(i + 1)
        rates.append(rate)
    plt.figure(dpi=200)
    plt.plot(epochs, rates)
    plt.plot(epochs, rates, 'r.')
    plt.xticks(epochs)
    plt.grid(True)
    plt.title(model_name + ' test on ' + dataset.name)
    plt.text(len(epochs)/2, max(rates)/2,
             'criterion=CrossEntropyLoss\noptimizer=Adam(lr=0.01)')
    plt.xlabel('epoch')
    plt.ylabel('accuracy (%)')
    plt.savefig(resultpath + '/' + model_name + '.png')
    plt.show()

#from ykModels import FullConnectedNetModel
#from ykDataloader import MnistDataLoader
#from ykModels import ConvolutionalNetModel
#from ykModels import InceptionConvolutionalNetModel
from ykModels import ResidualConvolutionalNetModel
from ykDataloader import Cifa10DataLoader

if __name__ == '__main__':
    cifa = Cifa10DataLoader(64, '../data/CIFA10', download=False)
    #cifa = MnistDataLoader(64, '../data/MNIST', download=False)
    img = np.array(cifa.trainImages[0][0])
    w = img.shape[0]
    h = img.shape[1]
    imgClassfierTraining(cifa,                                  # dataset
                         ResidualConvolutionalNetModel(3, w, h, 10),    # model
                         'cifa10-residential-net',                      # model name
                         20)                                    # epoch