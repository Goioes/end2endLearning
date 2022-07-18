# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 13:51:47 2021

@author: 20210124
"""
import os
import shutil
import numpy as np
import torch

def flow2rgb(flow_map, sparse, mask, max_value):
    if mask is not None: # for predictions
        h, w = np.shape(mask) # take flow map size
        flowMask = mask # mask for prediction, from target
        flow_map = torch.nn.functional.interpolate(flow_map.unsqueeze(0), (h,w), mode='bilinear', align_corners=False) # upsample prediction
        flow_map_np = flow_map.squeeze().detach().cpu().numpy()
        if sparse: # for KITTI
            flow_map_np[:,flowMask] = float('nan') # kiti make invalid pixels black
        else: # for Crowdflow
            flow_map_np[:,flowMask] = 0 # Crowdflow make background white
            
    else: # for targets
        _, h, w = np.shape(flow_map) # take flow map size
        flowMask = (flow_map[0,:,:] == 0) & (flow_map[1,:,:] == 0) # determine mask of target
        flow_map_np = flow_map.detach().cpu().numpy()
        if sparse: # for KITTI
            flow_map_np[:,flowMask] = float('nan') # kiti make invalid pixels black
        else: # for Crowdflow
            flow_map_np[:,flowMask] = 0 # Crowdflow make background white
        
    #flow_map_np[:,(flow_map_np[0] == 0) & (flow_map_np[1] == 0)] = 0
    #float('nan') # invalid pixels, no flow turn white (black original) should only be invalids in KITTI
    rgb_map = np.ones((3,h,w)).astype(np.float32) # rgb map has same size as flow map, but 3 values per pixel instead of 2
    if max_value is not None:
        normalized_flow_map = flow_map_np / max_value # rescale
    else:
        raise ValueError("""IF NO MAX_VALUE GIVEN, ALL VALUES BECOME NAN SINCE ZERO FLOW PIXELS ARE SET TO NAN(Line4),
                         SO NORMALIZATION DIVIDES BY NAN""")
    #    normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    rgb_map[0] += normalized_flow_map[0] # r value is 1 + vertical/horizontal flow?
    rgb_map[1] -= 0.5*(normalized_flow_map[0] + normalized_flow_map[1]) # g value is 1 - average of both
    rgb_map[2] += normalized_flow_map[1] # b value is 1 + vertical/horizontal flow?  
    return rgb_map.clip(0,1), flowMask

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return '{:.3f} ({:.3f})'.format(self.val, self.avg)
    
def saveCheckpoint(state, isBest, savePath, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(savePath,filename))
    if isBest:
        shutil.copyfile(os.path.join(savePath,filename), os.path.join(savePath,'model_best.pth.tar'))
        
def EPE(predictionFlowBatch, targetFlowBatch, sparse=False, mean=True):
    # take standard norm of flow vectors: dim=0 batch of flow images, dim=1 flow values, dim=2&3 frame dimensions
    EPE_map = torch.norm(targetFlowBatch-predictionFlowBatch,p=2,dim=1) 
    batch_size = EPE_map.size(0)
    if sparse:
        # invalid flow is defined with both flow coordinates to be exactly 0
        mask = (targetFlowBatch[:,0] == 0) & (targetFlowBatch[:,1] == 0) # returns tensor containing True/False 

        EPE_map = EPE_map[~mask] # removes all elements where mask is True --> invalid pixels
    if mean:
        return EPE_map.mean()
    else:
        return EPE_map.sum()/batch_size # used in multiscaleEPE?


def sparse_max_pool(input, size):
    '''Downsample the input by considering 0 values as invalid.

    Unfortunately, no generic interpolation mode can resize a sparse map correctly,
    the strategy here is to use max pooling for positive values and "min pooling"
    for negative values, the two results are then summed.
    This technique allows sparsity to be minized, contrary to nearest interpolation,
    which could potentially lose information for isolated data points.'''

    positive = (input > 0).float()
    negative = (input < 0).float()
    output = torch.nn.functional.adaptive_max_pool2d(input * positive, size) - torch.nn.functional.adaptive_max_pool2d(-input * negative, size)
    return output


def multiscaleEPE(network_output, target_flow, weights=None, sparse=False):
    def one_scale(output, target, sparse):

        b, _, h, w = output.size()
        if sparse:
            target_scaled = sparse_max_pool(target, (h, w)) # sample to size of layer output
        else:
            target_scaled = torch.nn.functional.interpolate(target, (h, w), mode='area') # downsample to size of layer output
        return EPE(output, target_scaled, sparse, mean=False)

    if type(network_output) not in [tuple, list]:
        network_output = [network_output]
    if weights is None:
        weights = [0.005, 0.01, 0.02, 0.08, 0.32]  # as in original article, quite arbitrary?
    assert(len(weights) == len(network_output))

    loss = 0
    for output, weight in zip(network_output, weights):
        loss += weight * one_scale(output, target_flow, sparse) # multiply errors at different layers in network by weights
    return loss


def realEPE(output, target, sparse=False):
    b, _, h, w = target.size() # batch, flows, height and width of frame
    # bilinear upsampling only for validation, train ouput is already upsampled using nearest
    upsampled_output = torch.nn.functional.interpolate(output, (h,w), mode='bilinear', align_corners=False)
    return EPE(upsampled_output, target, sparse, mean=True)
