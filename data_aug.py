import torch 
import random


def random_scale_2_stage(x, y):
    '''
    Input: circuit parameters & corresponding labels (Batch size x parameter length)
    Output: scaled parameters & labels               (Batch size x label length)

    Input range: (0,1)
    Scaled input range: (0,1)

    Output scaling: gain, cmrr, and ugf -- linear scaled
    '''
    bsz = len(x)
    scale = torch.rand(bsz,device=x.device).reshape(-1,1) 
    output_to_change = torch.randint(0,3,(bsz,))
    scaled_x = x.clone()
    scaled_y = y.clone()
    
    # ugf
    idx = torch.tensor([7,12])
    scaled_x[output_to_change==0][:,idx] *= scale[output_to_change==0].reshape(-1,1)
    # scaled_y[output_to_change==0] *= scale[output_to_change==0].reshape(-1,1)

    # gain
    idx = torch.tensor([6,7,10,11])
    scaled_x[output_to_change==1][:,idx] *= scale[output_to_change==1].reshape(-1,1)

    # cmrr
    idx = torch.tensor([6,7,9])
    scaled_x[output_to_change==2][:,idx] *= scale[output_to_change==2].reshape(-1,1)

    return scaled_x, scaled_y


def scale_2_stage(x, y, target):
    bsz = len(x)
    scale = 0.2 * torch.rand(bsz,device=x.device).reshape(-1,1) + 0.6
    scaled_x = x.clone()
    scaled_y = y.clone()
    if target == 'ugf':
        idx = torch.tensor([7,12]).to(x.device)
        scaled_x[:,idx] = scaled_x[:,idx] * scale
        # scaled_y = scaled_y * scale
    elif target == 'cmrr':
        idx = torch.tensor([6,7,9])
        scaled_x[:,idx] = scaled_x[:,idx] * scale
        # scaled_y = scaled_y #* scale
    elif target == 'gain':
        idx = torch.tensor([6,7,10,11])
        scaled_x[:,idx] = scaled_x[:,idx] * scale
        # scaled_y = scaled_y #* scale
    return scaled_x, scaled_y








