# This file gets moco embedding of given images
# Embedding from Jingbo's checkpoint 0003

# Current Usage: '--testmode True' gives test to the three mode
#                of getting embedding

# ToDo: 1. Confirm checkpoint 0003 with Jingbo's testing routines
#       2. Confirm the space as indeed linearly separable

# Repo used: 1. aihc-winter19-robustness
#            2. aihc-spring20-fewer

############# IMPORT SECTION ##################
import numpy as np
import pandas as pd
import argparse
import torch
from torchvision import models
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import sys
sys.path.append('/deep/u/akshaysm/aihc-spring20-fewer/moco/')
sys.path.append('/deep/u/akshaysm/aihc-winter19-robustness/chexpert-model')

from dataset.su_dataset import SUDataset
import moco.builder

############## ARGUMENTS FOR SUDATASET ########
parser = argparse.ArgumentParser(description='Predictor Tranform Image')
parser.add_argument('-a', metavar='ARCH', default='resnet18')
parser.add_argument('--moco-dim', type=int, default=128)
parser.add_argument('--moco-k', type=int, default=32768)
parser.add_argument('--moco-m', default=0.999, type=float)
parser.add_argument('--moco-t', default=0.07, type=float)
parser.add_argument('--img-size', dest='img_size', type=int, default=320)
parser.add_argument('--crop', dest='crop', type=int, default=320)
parser.add_argument('--rotate', dest='rotate', type=int, default=10)
parser.add_argument(
    '--maintain-ratio',
    dest='maintain_ratio',
    default=True,
    action='store_true')
parser.add_argument('--scale',dest='scale',type=int,default=320)
parser.add_argument('--clahe',dest='clahe',type=bool,default=False)
args = parser.parse_args()
args.normalization = 'cxr_norm'

############## MOCO CLASS ######################

class MoCoEmbedding:
    """This is a class that get embedding of CheXpert images.
       Required images to be tranformed and resized. Load
       checkpoints of Jingbo (epoch 3). May consider changing."""


    def __init__(self,mode='fc0'):
        """
        Initialize the MoCoEmbedding class

        Args:
            mode (string): This function load the encoder model into one of the following mode:
                           1. fc0 -> drop the last fc layer, use previous fc layer
                           2. fc1 -> use the last fc layer
                           3. conv -> drop both fc layers, use last conv layer
        """

        path = '/deep/group/aihc-bootcamp-spring2020/cxr_fewer_samples/experiments/jingbo/resnet18_mocov2_20200617-021146_SLURM1534372/'
        path += "checkpoint_0003.pth.tar"
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = dict((key[7:], value)
                          for (key, value) in checkpoint['state_dict'].items())
        model = moco.builder.MoCo(
            models.__dict__['resnet18'],

            #Parameter from Jingbo's checkpoint - Different from default!!
            K=49152,
            mlp=True,
            pretrained=False)
        model.load_state_dict(state_dict)
        if mode == 'fc1':
            self.model = model.encoder_q
        elif mode == 'fc0':

            #STRANGE: without flatten(), output size is [x, 512, 1, 1]
            self.model = torch.nn.Sequential(*list(model.encoder_q.children())[:-1],torch.nn.Flatten())
        else:
            self.model = torch.nn.Sequential(*list(model.encoder_q.children())[:-2],torch.nn.Flatten())
        self.model.eval()

    def getEmbedding(self,img):
        """Return the embeddings of a batch of images"""
        return self.model(img[0])


############## TESTING #########################

if __name__ == '__main__':

    our_task = {
        'No Finding': 0,
        'Enlarged Cardiomediastinum': 1,
        'Cardiomegaly': 2,
        'Lung Opacity': 3,
        'Consolidation': 4,
        'Pneumothorax': 5,
        'Pleural Effusion': 6}

    #STRANGE: output labels are not mapped correctly with our_task
    test = SUDataset(
        '/deep/group/CheXpert/',
        args,
        '/deep/group/CheXpert/CheXpert-v1.0/train.csv',
        False,
        our_task,
        False)

    our_dataloader = torch.utils.data.DataLoader(test,batch_size=5)

    for i, data in enumerate(our_dataloader):
        if i==0:
            print("Testing fc1..")
            MoCoTest1 = MoCoEmbedding('fc1')
            output = MoCoTest1.getEmbedding(data)
            print(output.shape)
        elif i==1:
            print("Testing fc0..")
            MoCoTest2 = MoCoEmbedding('fc0')
            output = MoCoTest2.getEmbedding(data)
            print(output.shape)
        elif i==2:
            print("Testing conv..")
            MoCoTest3 = MoCoEmbedding('conv')
            output = MoCoTest3.getEmbedding(data)
            print(output.shape)
        else:
            break
