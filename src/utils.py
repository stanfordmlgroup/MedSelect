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


def test_selector(selector, predictor, selector_ckpt_path, test_ld, k):
	"""Evaluate model on meta-test set.

	Args:
	    selector (nn.Module): the selector module
	    predictor (nn.Module): the predictor module
	    selector_ckpt_path (string): path to the selector checkpoint
	    test_ld (dataloader): the dataloader for meta-test set
	    k (int): the value of k with which the checkpoint was trained
	"""
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	if torch.cuda.device_count() > 0: #works even if only 1 GPU available
		print("Using", torch.cuda.device_count(), "GPUs!")
		selector = nn.DataParallel(selector)
		selector = selector.to(device)
		checkpoint = torch.load(selector_ckpt_path)
		selector.load_state_dict(checkpoint['model_state_dict'])
	else:
		print("Using CPU!")
		checkpoint = torch.load(selector_ckpt_path, map_location=torch.device('cpu'))
		new_state_dict = OrderedDict()
		for q, v in checkpoint['model_state_dict'].items():
			name = q[7:] # remove `module.`
			new_state_dict[name] = v
		selector.load_state_dict(new_state_dict)

	auroc_list, cond_list = evaluate(test_ld, selector, predictor, device, k, return_avg=False, numpy_sample = True)
	auroc = np.mean(auroc_list)
	cond_list = np.array(cond_list)
	print()
	print("Meta-test AUROC: %.4f" % auroc)

	cols = ['Condition', 'AUROC', 'n']
	auroc_dict = {'average': [auroc, 1000]}
	all_conds = list(set(cond_list))
	for cond in all_conds:
		mask = cond_list == cond
		a = np.mean(auroc_list[mask]) #average auroc for one condition
		b = np.sum(mask)
		auroc_dict[cond] = [a, b]
		print(cond + ": %.4f" % a)
	print()

	#write auroc to dataframe
	auroc_df = pd.DataFrame.from_dict(auroc_dict, orient='index', columns=['AUROC', 'n'])
	auroc_df.index.name = 'Condition'
	filename = selector_ckpt_path+"_auroc.csv"
	auroc_df.to_csv(filename)

	#compute distribution stats
	statistics = predictor.dist
	for key in statistics.keys():
		print("Mean for "+key+" is: %.4f"%(np.mean(np.array(statistics[key]))))
	stat_output = pd.DataFrame.from_dict(statistics)
	stat_output["cond"] = cond_list
	if len(stat_output["cond"].value_counts()) == 2:
		filename = selector_ckpt_path+"_holdout_stat.csv"
	else:
		filename = selector_ckpt_path+"_stat.csv"
	stat_output.to_csv(filename,index=False)
	print("Output written to file: "+filename+"\n")
#	predictor.clear()


if __name__ == '__main__':
	test_dset = RandomTaskDataset(positive_csv_path='/deep/group/activelearn/data/level1/meta_tst/holdout_positives.csv',
	 			      normal_csv_path='/deep/group/activelearn/data/level1/meta_tst/no_findings.csv',
				      unlabeled_pool_size=UNLABELED_POOL_SIZE,
                                      unlabeled_pos_frac=UNLABELED_POS_FRAC,
                                      query_set_size=QUERY_SET_SIZE,
                                      query_pos_frac=QUERY_POS_FRAC,
                                      conditions_used=HOLD_OUT,
                                      num_tasks=NUM_META_TEST_TASKS,
                                      deterministic=True)
	test_ld = torch.utils.data.DataLoader(dataset=test_dset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=4)
	selector = LSTMSelector(input_size=(USE_IMG*512 + USE_ASL*3))

	#predictor = meanPred(mode = 'cosine')
	#test_selector(selector, predictor, '/deep/group/activelearn/lstm_selector/k=200/ckpt/only_img/model_epoch1_iter90', test_ld, k=200)

	best_models = {10:'model_epoch5_iter30',
			20:'model_epoch5_iter150',
			40:'model_epoch5_iter30',
			80:'model_epoch4_iter150',
			100:'model_epoch5_iter90',
			200:'model_epoch1_iter90',
                        400:'model_epoch1_iter90'}
	for k in K:
		predictor = meanPred(mode = 'cosine')
		print(f"\n\nRunning for K={k}")
