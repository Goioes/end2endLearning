# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 13:23:32 2021

@author: 20210124
"""
import time
import torch

import trainUtils

def validate(valLoader, model, device, loss, epoch, printFreq):
    batchTime = trainUtils.AverageMeter()
    errors = trainUtils.AverageMeter()

    # switch to evaluate mode
    model.eval()
    
    startTime = time.time()
    for i, (input, target) in enumerate(valLoader):
        target = torch.cat(target,1).to(device)
        input = torch.cat(input,1).to(device)

        # compute output
        output = model(input)
        
        error = loss(output, target)
        
        # record EPE
        errors.update(error.item(), target.size(0))

        # measure elapsed time
        batchTime.update(time.time() - startTime)
        startTime = time.time()

        if i % printFreq == 0:
            print(f'Test: [{i}/{len(valLoader)}]\t Time {batchTime}\t MSE {errors}')

    print(f' * EPE {errors.avg}')
    return errors.avg
