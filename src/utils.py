from collections import OrderedDict
import numpy as np
import pandas as pd
from scipy.special import softmax
import scipy.stats
import torch
import torch.nn as nn
from datasets.random_task_dataset import RandomTaskDataset
from sklearn.metrics import roc_auc_score
from constants import *
from models.lstm_selector import LSTMSelector
from models.predictor import meanPred

def random_baseline(data, predictor, sampling_size):
	"""Compute the reward for the random baseline.

	Args:
		data (dict): a batch of tasks
		predictor (nn.Module): the predictor model
		sampling_size (int): how many X-rays to sample from pool

	Returns:
            res (Tensor): tensor of shape (batch_size) containing baseline reward for each task in the batch
        """
	selected = np.zeros(data['pool_labels'].shape)
	for k in range(data['pool'].shape[0]):
		selected_index = np.random.choice(range(UNLABELED_POOL_SIZE),sampling_size,replace = False)
		selected[k,selected_index] = 1
	data['mask'] = torch.Tensor(selected)
	data['pool'] = data['pool'].cpu()

	preds = predictor(data).squeeze(dim=2)
	res = np.array([0]*data['query_labels'].shape[0]).astype(float) #(batch_size,)
	for p in range(data['query_labels'].shape[0]):
		res[p] = roc_auc_score(data['query_labels'][p,:],preds[p,:])
	res = torch.Tensor(res)

	return res

def collate_fn(task_list):
	"""Custom collate function to batch tasks together into single tensors.

	Args:
	    task_list: a list of tasks, where each task is a dictionary as returned by the
                       __getitem__ method of the dataset object

	Returns:
	    batch: a dictionary with keys 'cond', 'pool', 'pool_labels', 'query', 'query_labels'.
		   The value of 'cond' is an array of shape (batch_size), the value of 'pool' is a
                   tensor of shape (batch_size, pool_size, embedding_size), the value of 'pool_labels'
                   is a tensor of shape (batch_size, pool_size), the value of 'query' is a tensor of
                   shape (batch_size, query_set_size, embedding_size), and the value of 'query_labels'
                   is a tensor of shape (batch_size, query_set_size)
	"""
	cond = [s['cond'] for s in task_list]
	batch = {'cond': cond, 'pool': None, 'pool_labels': None, 'query': None, 'query_labels': None}

	keys = ['pool', 'pool_labels', 'query', 'query_labels']
	for key in keys:
		tensor_list = [s[key] for s in task_list]
		stacked = torch.stack(tensor_list, dim=0)
		batch[key] = stacked

	batch['pool_labels'] = batch['pool_labels'].long()
	batch['query_labels'] = batch['query_labels'].long()
	return batch

def sample_once(weights, k):
	"""Sample from the logits output by a selector model.

		Args:
		    weights (Tensor): a tensor of shape (batch_size, pool_size) where each row is the
                                      output of the selector model for a single task
		    k (int): how many X-rays to choose for labeling from the pool

		Returns:
		    idx (np.ndarray): array of shape (batch_size, k) which contains indices of the X-rays
                                      in the pool that are selected
                    lob_prob (float): the log probability of the X-rays we sampled
	 """
	dist = torch.distributions.multinomial.Multinomial(total_count=k, logits=weights)
	x = dist.sample()
	log_prob = dist.log_prob(x)
	x = x.cpu().numpy().astype(int)

	idx = []
	for i in range(len(x)):
		idx.append(np.concatenate([[j]*x[i][j] for j in range(len(x[i]))]))
	idx = np.array(idx).astype(int)

	return idx, log_prob

def sample_once_numpy(weights, k):
	"""Sample from the logits output by a selector model, but using numpy ops which
           can be made deterministic easily. We don't care about log probs here.

	Args:
	    weights (Tensor): a tensor of shape (batch_size, pool_size) where each row is the
                              output of the selector model for a single task
	    k (int): how many X-rays to choose for labeling from the pool

	Returns:
	    idx (np.ndarray): array of shape (batch_size, k) which contains indices of the X-rays
                              in the pool that are selected
	"""
	np.random.seed(42) #for deterministic behavior
	weights = weights.to('cpu').numpy()

	#following two lines are needed due to softmax giving float32, causing rounding issues
	p_vals = softmax(weights, axis=1).astype('float64')
	p_vals = p_vals / np.sum(p_vals, axis=1)[:, np.newaxis]

	idx = []
	for i in range(weights.shape[0]):
		x = np.random.multinomial(k, p_vals[i]) #shape (pool_size,)
		idx.append(np.concatenate([[j]*x[j] for j in range(len(x))]))
	idx = np.array(idx).astype(int)
	return idx

def evaluate(val_ld, selector, predictor, device, k, return_avg=False, numpy_sample=False):
	"""Function to evaluate current model weights.

	Args:
	    val_ld (DataLoader): dataloader for the meta-val set
	    selector (nn.Module): the selector module
	    predictor (nn.Module): the predictor module
	    device (torch.device): device on which data should be
	    k (int): number of X-rays to sample from pool
            return_avg (bool): whether to return average auroc, or list of auroc's for each task
	    numpy_sample (bool): whether to use deterministic numpy sampling

	Return:
	    res (float): the mean of AUROC across all validation tasks if return_avg=True, OR
            res_list (np.ndarray): the AUROC scores for each task if return_avg=False.
	    Also returns a list of all conditions in the set
	"""
	print()
	print("Beginning validation epoch")

	ssidx = 0 if USE_IMG else 512  # Start Index of Pool for Selector
	seidx = 515 if USE_ASL else 512  # End Index of Pool for Selector

	was_training = selector.training
	selector.eval()
	res_list = []
	cond_list = []

	with torch.no_grad():
		for i, data in enumerate(val_ld):
			cond_list += data['cond']
			pool = data['pool'].cpu().numpy()
			pool_labels = data['pool_labels'].cpu().numpy()

			data['pool'] = data['pool'].to(device)
			logits = selector(data['pool'][:,:,ssidx:seidx])
			if numpy_sample: #deterministic sampling
				idx = sample_once_numpy(logits, k) #(batch_size, k)
			else:
				idx, log_prob = sample_once(logits, k) #(batch_size, k)

			selected = []
			selected_labels = []
			for p in range(len(idx)):
				selected.append(pool[p][idx[p]])
				selected_labels.append(pool_labels[p][idx[p]])

			selected = torch.Tensor(np.array(selected))
			selected_labels = torch.LongTensor(np.array(selected_labels))

			preds = predictor.forward_selected(selected, selected_labels, data['query'], data['query_labels'])
			preds = preds.squeeze(dim=2) #(batch_size, query_set_size)

			res = np.array([0]*data['query_labels'].shape[0]).astype(float)
			for p in range(data['query_labels'].shape[0]):
				res[p] = roc_auc_score(data['query_labels'][p,:],preds[p,:])
			res_list.append(res)

			if (i+1) % 10 == 0:
				print("Validation batch no.: ", i+1)

	if was_training:
		selector.train()

	res_list = np.concatenate(res_list)
	res = np.mean(res_list)

	if return_avg:
		return res, cond_list
	else:
		return res_list, cond_list

