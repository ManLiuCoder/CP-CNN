import os
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import utils
from trainer.test import validate
from models import get_model
from dataset import get_loader
from trainer import *
import argparse
import json

utils.set_seeding(0)
cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/test_cub.json')
config = parser.parse_args()
with open(config.config, 'r') as f:
    config.__dict__ = json.load(f)

model = get_model(config).cuda()
checkpoint = torch.load('results/cub/model_best.pth')
pretrained_dict = checkpoint['state_dict']
model_dict = model.state_dict()
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

val_loader  = get_loader(config, train=False, shuffle=False)['loader']


acc = validate(val_loader, model, config)
print('Results: {:.2f}'.format(acc))

