import pandas as pd
import numpy as np
import os
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class MultiTaskDataset(Dataset):
	"""Dataset which randomly samples two tasks."""

	def __init__(self, positive_csv_path, normal_csv_path, unlabeled_pool_size,
                     query_set_size, conditions_used, num_tasks, deterministic=False):
		""" 
		Initialize the Dataset object

		Args:
            	    positive_csv_path (string): path to csv containing X-rays that have at least one of the medical conditions
            	    normal_csv_path (string): path to csv containing X-rays that are positive for No Finding
            	    unlabeled_pool_size (int): number of X-rays to put inside one unlabeled pool which is fed into the selector
            	    query_set_size (int): number of X-rays used for testing the classifier model
            	    conditions_used (List[string]): list of all conditions that can be used by this Dataset object
            	    num_tasks (int): the total number of tasks that will be sampled
		    deterministic (bool): whether the task sampled is determined by the index fed in
		"""
		self.positive_csv = pd.read_csv(positive_csv_path)
		self.normal_csv = pd.read_csv(normal_csv_path).fillna(0)
		self.unlabeled_pool_size = unlabeled_pool_size
		self.query_set_size = query_set_size
		self.conditions_used = conditions_used
		self.num_tasks = num_tasks
		self.deterministic = deterministic

		self.positive_csv.fillna(0.0, inplace=True)
		self.positive_csv.replace(-1, 1.0, inplace=True) #U-ones

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
				 value of 'cond' is a list of 2 conditions. The value of 'pool' is a tensor of shape
                                 (pool_size, embedding_size) which contains sampled X-ray embeddings. The value of 'pool_labels'
                                 is the labels (0 or 1) for the X-rays in the pool, and is of shape (2, pool_size). The value of
                                 'query' is a tensor of shape (query_size, embedding_size) which contains sampled X-ray embeddings
                                 that do not overlap with the X-rays in the pool. The value of 'query_labels' is a tensor of shape
                                 (2, query_size) which contains the labels (0 or 1) for the X-rays in the query set. 
		"""

		if self.deterministic:
			np.random.seed(idx)

		task = {'cond': None, 'pool': None, 'pool_labels': None, 'query': None, 'query_labels': None}
		### CREATE UNLABELED POOL ###
		i = np.random.choice(len(self.conditions_used), size=(2), replace=False)
		cond = [self.conditions_used[i[0]], self.conditions_used[i[1]]]
		allowed_idx = (self.positive_csv[cond[0]] == 1.0) | (self.positive_csv[cond[1]] == 1.0)
		allowed_idx = np.argwhere(allowed_idx.tolist()).flatten()
		#allowed_idx = np.arange(len(self.positive_csv))

		embedding_list = []
		idx_list = []
		cond0_labels = []
		cond1_labels = []

		idx = np.random.choice(allowed_idx, size=int(0.75*self.unlabeled_pool_size), replace=False)
		embedding_list.append(self.positive_hdf5[idx])
		cond0_labels.append(self.positive_csv[cond[0]].iloc[idx].tolist())
		cond1_labels.append(self.positive_csv[cond[1]].iloc[idx].tolist())
		idx_list.append(idx)

		idx = np.random.choice(len(self.normal_csv), size=int(0.25*self.unlabeled_pool_size), replace=False)
		embedding_list.append(self.normal_hdf5[idx])
		cond0_labels.append(self.normal_csv[cond[0]].iloc[idx].tolist())
		cond1_labels.append(self.normal_csv[cond[1]].iloc[idx].tolist())
		idx_list.append(idx)

		pool = np.concatenate(embedding_list) #(pool_size, embedding_size)
		cond0_labels = np.concatenate(cond0_labels) #(pool_size,)
		cond1_labels = np.concatenate(cond1_labels) #(pool_size,)

		idx = np.random.permutation(np.arange(pool.shape[0])) #shuffle x-rays and labels
		pool = pool[idx]
		cond0_labels = cond0_labels[idx]
		cond1_labels = cond1_labels[idx]

		task['cond'] = cond
		task['pool'] = torch.Tensor(pool)
		task['pool_labels'] = torch.LongTensor(np.vstack((cond0_labels, cond1_labels)))

		pos_idx_remove = np.array(idx_list[0]) #positive csv X-rays used in pool
		neg_idx_remove = np.array(idx_list[1]) #normal csv X-rays used in pool

		### CREATE QUERY SET ###
		embedding_list = []
		cond0_labels = []
		cond1_labels = []

		remove_idx = set(pos_idx_remove)
		use_idx = list(set(allowed_idx).difference(remove_idx)) #remove X-rays used in pool
		idx = np.random.choice(use_idx, size=int(0.75*self.query_set_size), replace=False)
		embedding_list.append(self.positive_hdf5[idx])
		cond0_labels.append(self.positive_csv[cond[0]].iloc[idx].tolist())
		cond1_labels.append(self.positive_csv[cond[1]].iloc[idx].tolist())

		remove_idx = set(neg_idx_remove)
		use_idx = list(set(range(len(self.normal_csv))).difference(remove_idx)) #remove X-rays used in pool
		idx = np.random.choice(use_idx, size=int(0.25*self.query_set_size), replace=False)
		embedding_list.append(self.normal_hdf5[idx])
		cond0_labels.append(self.normal_csv[cond[0]].iloc[idx].tolist())
		cond1_labels.append(self.normal_csv[cond[1]].iloc[idx].tolist())

		query = np.concatenate(embedding_list) #(query_size, embedding_size)
		cond0_labels = np.concatenate(cond0_labels) #(pool_size,)
		cond1_labels = np.concatenate(cond1_labels) #(pool_size,)

		idx = np.random.permutation(np.arange(query.shape[0])) #shuffle x-rays and labels
		query = query[idx]
		cond0_labels = cond0_labels[idx]
		cond1_labels = cond1_labels[idx]

		task['query'] = torch.Tensor(query)
		task['query_labels'] = torch.LongTensor(np.vstack((cond0_labels, cond1_labels)))

		return task

if __name__ == '__main__':
	dset = MultiTaskDataset(positive_csv_path='/deep/group/activelearn/data/level1/meta_train/positives.csv', 
				 normal_csv_path='/deep/group/activelearn/data/level1/meta_train/no_findings.csv',
                	         unlabeled_pool_size=1000,
                         	 query_set_size=100,
                         	 conditions_used=["Cardiomegaly", "Consolidation"],
                         	 num_tasks=50,
                                 deterministic=True)
	res = dset[0]
	print(res['cond'])
	print(res['pool'].shape)
	print(res['pool_labels'].shape)
	print(res['query'].shape)
	print(res['query_labels'].shape)

	print(torch.sum(res['pool_labels'][0] * res['pool_labels'][1]))
	print(torch.sum(res['query_labels'][0] * res['query_labels'][1]))
