import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import scipy.ndimage as nd
from matplotlib import pyplot as plt
from torch import Tensor, einsum
from pdb import set_trace as bp
        

class Multi_BCELoss(nn.Module):
    def __init__(self, ignore_index=None, num_classes=3, **kwargs):
        super(Multi_BCELoss, self).__init__()
        self.kwargs = kwargs
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, predict, target, name, TEMPLATE):
        assert predict.shape[2:] == target.shape[2:], 'predict & target shape do not match'

        total_loss = []
        B = predict.shape[0]

        for b in range(B):
            # dataset_index = int(name[b][0:2])
            # if dataset_index == 10:
            #     template_key = name[b][0:2] + '_' + name[b][17:19]
            # elif dataset_index == 1:
            #     if int(name[b][-2:]) >= 60:
            #         template_key = '01_2'
            #     else:
            #         template_key = '01'
            # else:
            #     template_key = name[b][0:2]
            organ_list = TEMPLATE['01']
            for organ in organ_list:
                ce_loss = self.criterion(predict[b, organ-1], target[b, organ-1])
                total_loss.append(ce_loss)
        total_loss = torch.stack(total_loss)

        # print(name, total_loss, total_loss.sum()/total_loss.shape[0])

        return total_loss.sum()/total_loss.shape[0]
