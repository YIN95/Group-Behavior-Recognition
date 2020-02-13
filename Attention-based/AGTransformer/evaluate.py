import argparse
import gc
import logging
import os
import sys
import time

from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from utils import *
from loader import data_loader
from sklearn.model_selection import cross_val_score
from models import *
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
import statistics
import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt

from dataprocess import *
from models import *
from sklearn.metrics import confusion_matrix
from attrdict import AttrDict
from visualizer import vis, vissingle

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

parser = argparse.ArgumentParser()
# parser.add_argument('--model_path', default='model/group_test_0_with_model.pt',type=str)
parser.add_argument('--model_path', default='model/',type=str)
parser.add_argument('--sample_path', default='testgroup/', type=str)

def get_model(checkpoint):
    args = AttrDict(checkpoint['args'])
    model=GroupEncoder(
        embedding_dim=args.embedding_dim,
        h_dim=args.h_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_markers=37
    )
    model.load_state_dict(checkpoint['best_state'])
    model.cuda()
    model.eval()
    return model


def main(args):
    if os.path.isdir(args.model_path):
        filenames = os.listdir(args.model_path)
        filenames.sort()
        paths = [
            os.path.join(args.model_path, file_) for file_ in filenames
        ]
    else:
        paths = [args.model_path]

    accuracy=[]
    confusionmatrix=[]

    for ind, path in enumerate(paths):
        checkpoint = torch.load(path)
        model = get_model(checkpoint)
        _args = AttrDict(checkpoint['args'])
        _args.batch_size=40
        data_dset, loader = data_loader(_args, args.sample_path)
        with torch.no_grad():
            for batch in loader:
                (groupInput, target)=batch
                groupInput=groupInput.to(device)
                target=target.to(device)
                TAtt, BAtt, groupAttention,groupOutput=model(groupInput) # TAtt: 40X4X37X60, BAtt:40X4X37
                # frame=2
                # agent=3
                # timelist=[90,180,360]

                # bodyweights=BAtt[frame,agent,:]
                # bodyweights=bodyweights.cpu().numpy()
                # vissingle(ind,agent, bodyweights, timelist)


                data=TAtt[frame,agent,:,:]
                data=data.cpu().numpy()
                ax = sb.heatmap(data, vmin=0, vmax=0.022, cmap=sb.cubehelix_palette(n_colors=100, light=.95, dark=.08))
                # vissingle(ind,agent)
                plt.savefig('temporal/model_' + str(ind) +'frame_'+str(frame) + '__agentID__' + str(agent) + '.png')
                plt.close()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
