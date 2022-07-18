import os
import torch
from torch.utils.tensorboard import SummaryWriter

import dataLoader
import models
import trainer
import validate
import trainUtils

root = 'Data/'
valSize = 0.3
batchSize = 1
numWorkers = 1
shuffle = True
learningRate = 1e-3
momentum = 0.9
beta = 0.999
milestones = [100, 150, 200] # Epochs at which lr is divided by 2
gamma = 0.5
biasDecay = 0
weightDecay = 4e-4
trainLoader, valLoader = dataLoader.prepareDataLoaders()
networkData = None
epochs = 10
printFreq = 10
model = models.__dict__['baseModel'](networkData)
device, model, optimizer, scheduler = trainer.prepareTraining(model, biasDecay, weightDecay, learningRate, momentum, beta, milestones, gamma)
savePath = 'Results'

loss = torch.nn.MSELoss()
trainWriter = SummaryWriter(os.path.join(savePath, 'train'))
valWriter = SummaryWriter(os.path.join(savePath, 'val'))
bestError = -1
iteration = 0
print('Current average value and timings for single batch will be printed')
for epoch in range(epochs):
    trainError, iteration = trainer.training(trainLoader, trainWriter, model, optimizer, device, loss, epoch, iteration, printFreq)
    trainWriter.add_scalar('mean error', trainError, epoch)
    with torch.no_grad():
        currentError = validate.validate(valLoader, model, device, loss, epoch, printFreq)
    valWriter.add_scalar('mean error', currentError, epoch)

    scheduler.step
    if bestError < 0:
        bestError = currentError
    isBest = currentError < bestError
    bestError = min(currentError, bestError)
    trainUtils.saveCheckpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'bestError': bestError,
        }, isBest, savePath)
