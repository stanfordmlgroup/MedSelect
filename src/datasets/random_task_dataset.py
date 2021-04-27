import pandas as pd
import numpy as np
import os
import h5py
import torch
import torch.nn as nn
from constants import USE_ASL
from torch.utils.data import Dataset, DataLoader


MEAN_AGE = 60.33
STD_AGE = 17.92


class RandomTaskDataset(Dataset):
	"""Dataset which randomly samples a task."""

	def __init__(self, positive_csv_path, normal_csv_path, unlabeled_pool_size, 
                     unlabeled_pos_frac, query_set_size, query_pos_frac, conditions_used, 
                     num_tasks, deterministic=False, use_asl=False):
		""" 
		Initialize the Dataset object

		Args:
            	    positive_csv_path (string): path to csv containing X-rays that have at least one of the medical conditions
            	    normal_csv_path (string): path to csv containing X-rays that are positive for No Finding
            	    unlabeled_pool_size (int): number of X-rays to put inside one unlabeled pool which is fed into the selector
            	    unlabeled_pos_frac (float): fraction of the unlabeled pool X-rays which are positive for the condition
            	    query_set_size (int): number of X-rays used for testing the classifier model
            	    query_pos_frac (float): fraction of the query set X-rays which are positive for the condition
            	    conditions_used (List[string]): list of all conditions that can be used by this Dataset object
            	    num_tasks (int): the total number of tasks that will be sampled
		    deterministic (bool): whether the task sampled is determined by the index fed in
			use_asl (bool): whether to use age, sex, and laterality as feature
		"""
		self.positive_csv = pd.read_csv(positive_csv_path)
		self.normal_csv = pd.read_csv(normal_csv_path)
		self.unlabeled_pool_size = unlabeled_pool_size
		self.unlabeled_pos_frac = unlabeled_pos_frac
		self.query_set_size = query_set_size
		self.query_pos_frac = query_pos_frac
		self.conditions_used = conditions_used
		self.num_tasks = num_tasks
		self.deterministic = deterministic
		self.use_asl = use_asl

		pre, ext = os.path.splitext(positive_csv_path)
		positive_hdf5_path = pre + ".hdf5"
		pre, ext = os.path.splitext(normal_csv_path)
		normal_hdf5_path = pre + ".hdf5"

		self.positive_hdf5 = h5py.File(positive_hdf5_path, "r").get("dataset")[:, :]
		self.normal_hdf5 = h5py.File(normal_hdf5_path, "r").get("dataset")[:, :]

		assert(self.positive_hdf5.shape[0] == len(self.positive_csv))
		assert(self.normal_hdf5.shape[0] == len(self.normal_csv))

	def __len__(self):
		"""Return the length of the dataset"""
		return self.num_tasks

	def __getitem__(self, idx):
		"""Get the element of the dataset with index idx. We will just randomly sample a task, and return that.

		Args:
		    idx (int): integer index of the item to fetch

		Returns:
		    task (dict): a dictionary with keys 'pool', 'pool_labels', 'query', 'query_labels' and 'cond'. The 
				 value of 'cond' is the condition in the task. The value of 'pool' is a tensor of shape
                                 (pool_size, embedding_size) which contains sampled X-ray embeddings. The value of 'pool_labels'
                                 is the labels (0 or 1) for the X-rays in the pool, and is of shape (pool_size). The value of
                                 'query' is a tensor of shape (query_size, embedding_size) which contains sampled X-ray embeddings
                                 that do not overlap with the X-rays in the pool. The value of 'query_labels' is a tensor of shape
                                 (query_size) which contains the labels (0 or 1) for the X-rays in the query set. 
		"""
		def get_age_sex_lat(df, idx):
			age_sex_lat_df = df.loc[idx, ['Sex', 'Age', 'Frontal/Lateral']]
			age_sex_lat_df['Sex'] = (age_sex_lat_df['Sex'] == 'Female').astype(np.float32)
			age_sex_lat_df['Age'] = ((age_sex_lat_df['Age'] - MEAN_AGE) / STD_AGE).astype(np.float32)
			age_sex_lat_df['Frontal/Lateral'] = (age_sex_lat_df['Frontal/Lateral'] == 'Frontal').astype(np.float32)
			return age_sex_lat_df.values

		if self.deterministic:
			np.random.seed(idx)

		### CREATE UNLABELED POOL ###
		i = np.random.choice(len(self.conditions_used))
		cond = self.conditions_used[i]

		mask = (self.positive_csv[cond] == 1.0).tolist()
		pos_idx = np.argwhere(mask).flatten()
		num_pos = int(np.ceil(self.unlabeled_pool_size * self.unlabeled_pos_frac))
		pos_idx = np.random.choice(pos_idx, size=(num_pos), replace=False)
		pos_embedding = self.positive_hdf5[pos_idx] #(num_pos, embedding_size)
		pos_labels = np.ones([num_pos])

		# Age, Sex, Laterality
		pos_embedding = np.concatenate((pos_embedding, get_age_sex_lat(self.positive_csv, pos_idx)), axis=1) if self.use_asl else pos_embedding

		num_neg = self.unlabeled_pool_size - num_pos
		neg_idx = np.random.choice(len(self.normal_csv), size=(num_neg), replace=False)
		neg_embedding = self.normal_hdf5[neg_idx] #(num_neg, embedding_size)
		neg_labels = np.zeros([num_neg])

		# Age, Sex, Laterality
		neg_embedding = np.concatenate((neg_embedding, get_age_sex_lat(self.normal_csv, neg_idx)), axis=1) if self.use_asl else neg_embedding

		pool = np.concatenate([pos_embedding, neg_embedding], axis=0) #(pool_size, embedding_size)
		pool_labels = np.concatenate([pos_labels, neg_labels], axis=0) #(pool_size)

		idx = np.random.permutation(np.arange(pool.shape[0])) #shuffle x-rays and labels
		pool = pool[idx]
		pool_labels = pool_labels[idx]

		### CREATE QUERY SET ###
		mask = (self.positive_csv[cond] == 1.0).tolist()
		idx = np.argwhere(mask).flatten()
		idx = set(idx).difference(set(pos_idx)) #remove already sampled X-rays that were in unlabeled pool
		idx = list(idx)

		num_pos = int(np.ceil(self.query_set_size * self.query_pos_frac))
		idx = np.random.choice(idx, size=(num_pos), replace=False)
		pos_embedding = self.positive_hdf5[idx] #(num_pos, embedding_size)
		pos_labels = np.ones([num_pos])

		# Age, Sex, Laterality
		pos_embedding = np.concatenate((pos_embedding, get_age_sex_lat(self.positive_csv, idx)), axis=1) if self.use_asl else pos_embedding

		num_neg = self.query_set_size - num_pos
		idx = range(len(self.normal_csv))
		idx = set(idx).difference(set(neg_idx)) #remove already sampled X-rays that were in unlabeled pool
		idx = list(idx)

		idx = np.random.choice(idx, size=(num_neg), replace=False)
		neg_embedding = self.normal_hdf5[idx] #(num_neg, embedding_size)
		neg_labels = np.zeros([num_neg])

		# Age, Sex, Laterality
		neg_embedding = np.concatenate((neg_embedding, get_age_sex_lat(self.normal_csv, idx)), axis=1) if self.use_asl else neg_embedding

		query = np.concatenate([pos_embedding, neg_embedding], axis=0) #(query_set_size, embedding_size)
		query_labels = np.concatenate([pos_labels, neg_labels], axis=0) #(query_set_size)

		idx = np.random.permutation(np.arange(query.shape[0])) #shuffle x-rays and labels
		query = query[idx]
		query_labels = query_labels[idx]

		task = {'pool': torch.Tensor(pool), 
                        'pool_labels': torch.LongTensor(pool_labels), 
                        'query': torch.Tensor(query), 
                        'query_labels': torch.LongTensor(query_labels),
			'cond': cond}
		return task

if __name__ == '__main__':
	dset = RandomTaskDataset(positive_csv_path='/deep/group/activelearn/data/level1/meta_tst/holdout_positives.csv', 
				 normal_csv_path='/deep/group/activelearn/data/level1/meta_tst/no_findings.csv',
                	         unlabeled_pool_size=1000,
                        	 unlabeled_pos_frac=0.5,
                         	 query_set_size=100,
                         	 query_pos_frac=0.7,
                         	 conditions_used=["Edema", "Atelectasis"],
                         	 num_tasks=50,
				 use_asl=USE_ASL)
	res = dset[0]
	print(res['cond'])
	print(res['pool'].shape)
	print(res['pool_labels'].shape)
	print(res['query'].shape)
	print(res['query_labels'].shape)

	print()
	print(torch.sum(res['pool_labels']))
	print(torch.sum(res['query_labels']))
