import torch
import torch.nn as nn
from torch.autograd import Variable as V
import torch.nn.functional as F

class cloud_loss(nn.Module):
    def __init__(self, batch=True):
        super(cloud_loss, self).__init__()
        self.batch = batch
        self.loss = nn.L1Loss(reduction='sum')
    
    def __call__(self, pred, target):
        size = pred.shape[0] * pred.shape[1]
        cloud, clean, mask = target
        pred_mask  = pred * mask # cloud part
        clean_mask = clean * mask
        cloud_loss =  self.loss(pred_mask, clean_mask)
        
        pred_not_mask = pred * (1-mask)
        cloud_not_mask = cloud * (1-mask)
        clean_loss = self.loss(pred_not_mask, cloud_not_mask)
        
        reg_loss = self.loss(pred, clean)        
        
        return (cloud_loss + clean_loss + reg_loss) / cloud.shape.numel()
        