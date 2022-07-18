import time 
import torch

import dataLoader
import trainUtils

def training(trainLoader, trainWriter, model, optimizer, device, loss, epoch, iteration, printFreq):
    batchTime = trainUtils.AverageMeter()
    dataLoadTime = trainUtils.AverageMeter()
    errors = trainUtils.AverageMeter()

    # switch to train mode
    model.train()
    startTime = time.time()
    for i, (input, target) in enumerate(trainLoader):
        # measure data loading time
        dataLoadTime.update(time.time() - startTime)

        input = torch.cat(input,1).to(device)
        target = torch.cat(target,1).to(device)

        # compute output
        output = model(input) 
        
        # record loss and EPE
        error = loss(output, target)
        trainWriter.add_scalar('trainLoss', error.item(), iteration)
        errors.update(error.item(), target.size(0))

        # compute gradient and do optimization step
        optimizer.zero_grad()
        error.backward()
        optimizer.step()
        
        # measure elapsed time
        batchTime.update(time.time() - startTime)
        startTime = time.time()
        
        iteration +=1
        if i % printFreq == 0:
            print(f'Epoch: [{epoch}][{i}/{len(trainLoader)}]\t SingleBatchTime {batchTime}\t DataLoadTime {dataLoadTime}\t MSE {errors}')

    return errors.avg, iteration

def prepareTraining(model, biasDecay, weightDecay, learningRate, momentum, beta, milestones, gamma):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    #print(model.__dict__)
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate, betas=(momentum, beta))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    return device, model, optimizer, scheduler
