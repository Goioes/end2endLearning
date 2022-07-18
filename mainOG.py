# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 13:17:48 2021

@author: 20210124
"""

import sys
#print(sys.path.append('C:\\Users\\20210124\\Documents\\AdvancedSensingDL\\Code\\5aua0-2020-group-10-project-2\CodeGCP'))
import os
import torch
import datetime
import time
from torch.utils.tensorboard import SummaryWriter
import shutil
import glob

import dataLoader
import train
import validate
import trainUtils
import models

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Hyperparameters:
settings = {}
settings['pretrained'] = False
if settings['pretrained']:
    settings['epochsPreTrained'] = 5
else:
    settings['epochsPreTrained'] = 0
settings['arch'] = 'flownets'
#settings['arch'] = 'flownet_with_pwc'
settings['epochsTotal'] = 20
settings['solver'] = 'adam'
settings['workers'] = 1
#settings['epochSize'] = 1000
settings['batchSize'] = 1
settings['learningRate'] = 0.0001
settings['momentum'] = 0.9
settings['beta'] = 0.999
settings['weightDecay'] = 4e-4
settings['biasDecay'] = 0
settings['divFlow'] = 20
#settings['multiscaleWeights'] = [0.005,0.01,0.02,0.08,0.32]
settings['milestones'] = [100,150,200] # epochs at which lr is divided by 2
settings['printFreq'] = 10

settings['valSize'] = 0.25
settings['shuffle'] = True
settings['augmented'] = False
settings['dataset'] = 'CrowdFlow' # Other option is 'Kitti2015'
settings['val_writers'] = '' # Set in initialization
settings['save_path'] = '' # Set in initialization
settings['train_writer'] = '' # Set in initialization
settings['val_writers'] = '' # Set in initialization

def run(dataLoaders, model, optimizer, scheduler, device, settings):
    best_EPE = -1
    n_iter = 0
    print('Prints current value (average value), timings for single batch')
    for epoch in range(settings['epochsPreTrained'], settings['epochsTotal']):
        # train for one epoch
        train_EPE, n_iter = train.training(dataLoaders['train'], model, optimizer, device, settings, epoch, n_iter)
        settings['train_writer'].add_scalar('mean EPE', train_EPE, epoch)
        # evaluate on validation set
        with torch.no_grad():
            currentEPE = validate.validate(dataLoaders['val'], model, device, settings, epoch)
        settings['val_writers']['EPE'].add_scalar('mean EPE', currentEPE, epoch)
    
        scheduler.step()
        if best_EPE < 0:
            best_EPE = currentEPE
    
        is_best = currentEPE < best_EPE
        best_EPE = min(currentEPE, best_EPE)
        trainUtils.save_checkpoint({
                'epoch': epoch + 1,
                'arch': settings['arch'] ,
                'state_dict': model.state_dict(),
                'best_EPE': best_EPE,
                'div_flow': settings['divFlow']
            }, is_best, settings['save_path'])

def initDataLoading(settings):
    # LOAD DATA
    if settings['dataset'] == 'Kitti2015':
        root = r'C:\Users\20210124\Documents\AdvancedSensingDL\Data\Kitti2015\Flow\training'
        listPathSamples = dataLoader.findKITTIImagePaths(root)
    elif settings['dataset'] == 'CrowdFlow':
        root = r'/home/teunurselmann/TUBCrowdFlow'
        listPathSamples = dataLoader.findCrowdFlowImagePaths(root, scenetype='static') #scenetype can be 'all', 'static' or 'dynamic'
    samples = dataLoader.splitData(listPathSamples, settings['valSize']) # Dictionary containing training and validation set
    dataSets = {mode: dataLoader.ListDataset(samples[mode], mode, settings['dataset'], settings['divFlow'],
                                             augmented=settings['augmented']) 
                for mode in ['train', 'val']} # Create torch Dataset instance
    dataLoaders = {mode: torch.utils.data.DataLoader(dataSets[mode], batch_size=settings['batchSize'],
                                                     shuffle=settings['shuffle'], num_workers=settings['workers']) 
                  for mode in ['train', 'val']} # Create torch Dataloader instances
    return dataLoaders

def initTrainSettings(settings):
    # SETTINGS TO TRAIN
    if settings['pretrained']:
        pretrainedPath = r'/home/teunurselmann/5aua0-2020-group-10-project-2/CodeGCP/LogResults/CrowdFlow/06-15-12_10/flownets_adam_b4_lr0.0001/model_best.pth.tar'
        network_data = torch.load(pretrainedPath)
        settings['arch'] = network_data['arch']
        model = models.__dict__[settings['arch']](network_data['state_dict'])
        print(f'=> using pre-trained model {settings["arch"]}')
    else:
        network_data = None
        model = models.__dict__[settings['arch']](network_data)
        print(f'=> creating model {settings["arch"]}')


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    param_groups = [{'params': model.bias_parameters(), 'weight_decay': settings['biasDecay']},
                        {'params': model.weight_parameters(), 'weight_decay': settings['weightDecay']}]
    optimizer = torch.optim.Adam(param_groups, settings['learningRate'], 
                                             betas=(settings['momentum'], settings['beta']))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings['milestones'], gamma=0.5)
    return device, model, optimizer, scheduler

def initLogResults(settings):
    # SETTINGS TO SAVE RESULTS
    save_path = f"{settings['arch']}_{settings['solver']}_b{settings['batchSize']}_lr{settings['learningRate']}"
    timestamp = datetime.datetime.now().strftime("%m-%d-%H_%M")
    settings['save_path'] = os.path.join('LogResults', settings['dataset'], timestamp, save_path)
    print(f'=> will save everything to {settings["save_path"]}')
    if not os.path.exists(settings['save_path']):
        os.makedirs(settings['save_path'])
    
        # EPE's during training and validating
    settings['train_writer'] = SummaryWriter(os.path.join(settings['save_path'],'train'))
    settings['val_writers'] = {}
    settings['val_writers']['EPE'] = SummaryWriter(os.path.join(settings['save_path'],'val'))
        # write groundtruth and estimated flow images in directory 1 of val, corresponding input image 1 and 2 in directory 2 and 3
    settings['val_writers']['Flow'] = SummaryWriter(os.path.join(settings['save_path'],'val','Flow'))
    settings['val_writers']['InputImage1'] = SummaryWriter(os.path.join(settings['save_path'],'val','InputImage1'))
    settings['val_writers']['InputImage2'] = SummaryWriter(os.path.join(settings['save_path'],'val','InputImage2'))
    
if __name__ == '__main__':
    dataLoaders = initDataLoading(settings)
    device, model, optimizer, scheduler = initTrainSettings(settings)
    initLogResults(settings)
    run(dataLoaders, model, optimizer, scheduler, device, settings)