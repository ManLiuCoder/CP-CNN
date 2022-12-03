import os
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils import RunningMean, accuracy
import utils



def update_dest(acc_meter, key, content, target):
    if not key in acc_meter.keys():
        acc_meter[key] = RunningMean()
    acc_tmp = accuracy(content, target, topk=(1,))
    acc_meter[key].update(acc_tmp[0], len(target))


def validate(val_loader, model, cfg=None):
    acc_meter = {}
    model = model.eval()

    with torch.no_grad():
        for idx, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda(non_blocking=True)
            logits_list = model(input)
            # ------update acc meter------
            update_dest(acc_meter, "Accuracy", logits_list, target)
            
        return acc_meter['Accuracy'].value


