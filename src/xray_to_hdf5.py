import numpy as np
import pandas as pd
import argparse
import h5py
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from tqdm import tqdm
import moco.builder as builder
from moco.su_dataset import SUDataset

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
parser.add_argument('--testmode',dest='testmode',type=bool,default=False)
args = parser.parse_args()
args.normalization = 'cxr_norm'

############## MOCO CLASS ######################

#ToDO: consider making this a subsection of dataloader/dataset
class MoCoEmbedding:
    '''This is a class that get embedding of CheXpert images.
       Required images to be tranformed and resized. Load
       checkpoints of Jingbo (epoch 3). May consider changing.'''

    def __init__(self,mode='fc0'):
        '''This function load the encoder model into a required
           mode (pass in strings):
           1. fc0 -> drop the last fc layer, use previous fc layer
           2. fc1 -> use the last fc layer
           3. conv -> drop both fc layers, use last conv layer'''

        path = '/deep/group/aihc-bootcamp-spring2020/cxr_fewer_samples/experiments/jingbo/resnet18_mocov2_20200617-021146_SLURM1534372/'
        path += "checkpoint_0003.pth.tar"
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = dict((key[7:], value)
                          for (key, value) in checkpoint['state_dict'].items())
        model = builder.MoCo(
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
        '''This function return the embedding of a batch of images'''
        return self.model(img[0])

def xray_to_hdf5(csv_path, out_path):
	"""Convert each Xray in the csv into an embedding, and store them as hdf5.
	
	Args:
	    csv_path (string): path to csv containing all the X-rays
	    out_path (string): path to desired output file 
	"""
	our_task = {'No Finding': 0,
	            'Enlarged Cardiomediastinum': 1,
		    'Cardiomegaly': 2,
		    'Lung Opacity': 3,
                    'Lung Lesion': 4,
                    'Edema': 5,
                    'Consolidation': 6,
                    'Pneumonia': 7,
                    'Atelectasis': 8}
	test = SUDataset(
        	'',
        	args,
        	csv_path,
        	False,
        	our_task,
        	False)

	print(test.__len__())
	our_dataloader = torch.utils.data.DataLoader(test,batch_size=128,num_workers=6)

	all_batches = []
	moco = MoCoEmbedding('fc0')
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	moco.model.eval()
	moco.model.to(device)

	with torch.no_grad():
		for i, data in enumerate(tqdm(our_dataloader)):
			data[0] = data[0].to(device)
			out = moco.getEmbedding(data)
			all_batches.append(out)

		all_batches = torch.cat(all_batches, dim=0)
		print("Final output shape is:")
		print(all_batches.shape)

	print("Writing to hdf5")
	file = h5py.File(out_path, "w")
	dataset = file.create_dataset("dataset", data=all_batches.to('cpu').numpy())
	

if __name__ == '__main__':
	xray_to_hdf5('/deep/group/activelearn/data/mimic-cxr/non_holdout_positives.csv',
		     '/deep/u/akshaysm/temp.hdf5')

