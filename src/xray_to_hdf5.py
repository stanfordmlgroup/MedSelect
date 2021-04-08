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
args = argparse.Namespace()
args.a='resnet18'
args.clahe=False
args.crop=320
args.img_size=320
args.maintain_ratio=True
args.moco_dim=128
args.moco_k=32768
args.moco_m=0.999
args.moco_t=0.07
args.normalization='cxr_norm'
args.rotate=10
args.scale=320
args.testmode=False

############## MOCO CLASS ######################

#ToDO: consider making this a subsection of dataloader/dataset
class MoCoEmbedding:
    '''This is a class that get embedding of CheXpert images.
       Required images to be tranformed and resized. Load
       checkpoints of Jingbo (epoch 3). May consider changing.'''

    def __init__(self,mode, path):
        '''This function load the encoder model into a required
           mode (pass in strings):
           1. fc0 -> drop the last fc layer, use previous fc layer
           2. fc1 -> use the last fc layer
           3. conv -> drop both fc layers, use last conv layer'''

        #path = '/deep/group/aihc-bootcamp-spring2020/cxr_fewer_samples/experiments/jingbo/resnet18_mocov2_20200617-021146_SLURM1534372/'
        #path += "checkpoint_0003.pth.tar"
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = dict((key[7:], value)
                          for (key, value) in checkpoint['state_dict'].items())
        model = builder.MoCo(
            models.__dict__['resnet18'],

            #Parameter from MoCo checkpoint - Different from default!!
            K=49152,
            mlp=True,
            pretrained=False)
        model.load_state_dict(state_dict)
        if mode == 'fc1':
            self.model = model.encoder_q
        elif mode == 'fc0':

            self.model = torch.nn.Sequential(*list(model.encoder_q.children())[:-1],torch.nn.Flatten())
        else:
            self.model = torch.nn.Sequential(*list(model.encoder_q.children())[:-2],torch.nn.Flatten())
        self.model.eval()

    def getEmbedding(self,img):
        '''This function return the embedding of a batch of images'''
        return self.model(img[0])

def xray_to_hdf5(csv_path, out_path, ckpt_path):
	"""Convert each Xray in the csv into an embedding, and store them as hdf5.
	
	Args:
	    csv_path (string): path to csv containing all the X-rays
	    out_path (string): path to desired output file
            ckpt_path (string): path to MoCo checkpoint
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
	moco = MoCoEmbedding('fc0', ckpt_path)
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
	prs = argparse.ArgumentParser(description='Get MoCo embeddings from X-ray images listed in a csv file and store them in hdf5 format')
	prs.add_argument('-d', '--data', type=str, nargs='?', required=True,
			    help='path to csv containing X-ray file paths. The paths to the images should be under the \"Path\" column')
	prs.add_argument('-o', '--output_path', type=str, nargs='?', required=True, help='path to intended output hdf5 file)')
	prs.add_argument('-c', '--checkpoint_path', type=str, nargs='?', required=True, help='path to MoCo checkpoint')
	arguments = prs.parse_args()
	csv_path = arguments.data
	out_path = arguments.output_path
	ckpt_path = arguments.checkpoint_path

	xray_to_hdf5(csv_path, out_path, ckpt_path)

	#xray_to_hdf5('/deep/group/activelearn/data/mimic-cxr/non_holdout_positives.csv',
	#	     '/deep/u/akshaysm/temp.hdf5')

