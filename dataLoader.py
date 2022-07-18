# -*- coding: utf-8 -*-
"""
@author: 20210124
"""

import os
import glob
import numpy as np
import cv2
from imageio import imread
import torch
import random
from torchvision import datasets, transforms
import csv
import pandas as pd

class ListDataset(torch.utils.data.Dataset):
    def __init__(self, samplesPathList):
        self.samplesPathList = samplesPathList

    def loader(self, imagePaths): 
        images = [imread(imagePath).astype(np.float32)[:,:,:3]/255 for imagePath in imagePaths]
        return images
        
    def __getitem__(self, index):
        inputPaths = self.samplesPathList[index][0]
        targets = self.samplesPathList[index][1]
        inputs = self.loader(inputPaths)
        inputs = dataNormalization(inputs)
        return inputs, targets

    def __len__(self):
        return len(self.samplesPathList)

def findImagePaths(root):
    listPathSamples = []
    controlsDF = pd.read_csv(os.path.join(root, 'controls.csv'))
    for frontImage in glob.iglob(os.path.join(root, 'FrontImages', '*.png')): # Find all front images
        imageName = os.path.basename(frontImage)
        leftImage = os.path.join(root, 'LeftImages', imageName) # Find corresponding left image
        rightImage = os.path.join(root, 'RightImages', imageName) # Find corresponding right image
        stepID = int(imageName[5:-4]) # Extract step identifier
        throttle = controlsDF.loc[controlsDF['step'] == stepID, 'throttle'].values.astype(np.float32)
        steer = controlsDF.loc[controlsDF['step'] == stepID, 'steer'].values.astype(np.float32)
        listPathSamples.append([[frontImage, leftImage, rightImage], [throttle, steer]]) # Create list of input images and corresponding controls 
    return listPathSamples

def splitData(listPathSamples, valSize):
    valIdx = np.random.choice(len(listPathSamples), int(len(listPathSamples)*valSize), replace=False)
    val, train = [], []
    for i in range(len(listPathSamples)):
        if i in valIdx:
            val.append(listPathSamples[i])
        else:
            train.append(listPathSamples[i])
    samples = {'train': train, 'val': val}
    return samples

def dataNormalization(inputs):
     inputsNormalized = [] 
     for image in inputs:
         inputsNormalized.append(
            transforms.Compose([
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.410,0.387,0.406], std=[0.116,0.141,0.134]) # mean and std refer to original channel values of entire dataset 
                ])(image)
            )
     return inputsNormalized

def prepareDataLoaders(root='Data/', valSize=0.3, batchSize=1, numWorkers=1, shuffle=True):
    listPathSamples = findImagePaths(root)
    samples = splitData(listPathSamples, valSize)
    print(f"Train data: {samples['train'][0]}")
    print(f"Val data: {samples['val'][0]}")
    dataSets = {mode: ListDataset(samples[mode]) for mode in ['train', 'val']} # Create torch Datasets
    dataLoaders = {mode: torch.utils.data.DataLoader(dataSets[mode], batch_size=batchSize,
        shuffle=shuffle, num_workers=numWorkers) for mode in ['train', 'val']} # Create torch DataLoaders 
    return dataLoaders['train'], dataLoaders['val']

if __name__ == '__main__':
    trainLoader, valLoader = prepareDataLoaders()
