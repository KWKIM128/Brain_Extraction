import torch
import torch.nn as nn
    
def dice_metrics(inputs, targets, smooth=1e-8):
    
    inputs = (inputs>=0.5)
    targets = (targets>=0.5)

    intersection = (inputs * targets).sum()
    dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

    return dice

def precision_metrics(inputs, targets, smooth=1e-8):
    inputs = (inputs>=0.5)
    targets = (targets>=0.5)

    intersection = (inputs * targets).sum()
    
    precision = (intersection + smooth)/(inputs.sum() + smooth)
    
    return precision
    
def recall_metrics(inputs, targets, smooth=1e-8):
    inputs = (inputs>=0.5)
    targets = (targets>=0.5)

    intersection = (inputs * targets).sum()
    
    recall = (intersection + smooth)/(targets.sum() + smooth)
    
    return recall    

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-8):

        inputs = torch.sigmoid(inputs)
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        dice_loss = 1 - dice

        return dice_loss