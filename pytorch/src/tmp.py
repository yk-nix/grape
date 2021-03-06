# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 11:27:00 2021

@author: Administrator
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.5)  # pause a bit so that plots are updated

def faces_test():
    landmarks_frame = pd.read_csv('data/faces/face_landmarks.csv')
    for i in range(len(landmarks_frame)):
        img_name = landmarks_frame.iloc[i, 0]
        landmarks = landmarks_frame.iloc[i, 1:]
        landmarks = np.asarray(landmarks)
        landmarks = landmarks.astype('float').reshape(-1, 2)
        plt.figure()
        show_landmarks(io.imread(os.path.join('data/faces/', img_name)),
                       landmarks)
        plt.show()


if __name__ == '__main__':
    x = np.arange(1, 11, 1)
    y = 0.23 * x
    plt.plot(x, y)
    plt.plot(x, y, 'r.')
    plt.grid(True)
    plt.title('residential net test on CIFA10')
    plt.text(6, 1,'criterion=CrossEntropyLoss\noptimizer=SGD(lr=0.01,momentum=0.9)')
    l =[1,2,3,5]
    print(int(max(l)/2))
    plt.show()
    pass
