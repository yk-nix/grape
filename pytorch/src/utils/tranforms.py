# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 21:51:00 2021

@author: yoka
"""
import random
import torchvision.transforms as T

class Compse(object):
    '''
    transforms(List): list of operations to carry out transformations
    '''
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target
    
class ToTensor(object):
    '''
    img(PIL.Image): calling torchvision.transforms.ToTensor to 
    Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    target: it will not be changed
    '''
    def __call__(self, img, target):
        img = T.ToTensor(img)
        return img, target

class RandomHorizontalFlip(object):
    '''
    probility(float): the probability to flip the image
    '''
    def __init__(self, probability):
        self.prob = probability
    
    def __call__(self, img, target):
        if random.random() < self.prob:
            img = img.flip(-1)
            width = img.shape[-1]
            bbox = target['boxes']
            bbox[:,[0,2]] = width - bbox[:,[2,0]]
            target['boxes'] = bbox
        return img, target