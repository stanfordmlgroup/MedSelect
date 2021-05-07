import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from utils import sample_once, collate_fn, evaluate, random_baseline
from sklearn.metrics import roc_auc_score
from constants import *
from datasets.random_task_dataset import RandomTaskDataset
from models.lstm_selector import LSTMSelector
from models.predictor import meanPred

def train_model(train_ld, val_ld, predictor, selector, save_path, num_epochs, lr, k):
	"""Train the selector model.

	Args:
	    train_ld (dataloader): dataloader for the meta-train set
	    val_ld (dataloader): dataloader for the meta-val set
	    predictor (nn.Module): the predictor module
	    selector (nn.Module): the trainable selector module
	    save_path (string): path to save checkpoints
	    num_epochs (int): number of epochs
	    lr (float): learning rate
	    k (int): number of X-rays to select from pool
	"""
	optimizer = torch.optim.Adam(selector.parameters(), lr=lr)
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	selector = selector.to(device)
	selector.train()

	ssidx = 0
	seidx = 515 if USE_ASL else 512  # End Index of Pool for Selector

	begin_time = time.time()
	report_examples = 0
	report_reward = 0
	best_metric = None
	print('begin selector training K=%d' % k)
	for epoch in range(num_epochs):
		if epoch == 2: #multiply learning rate by 0.1
			for g in optimizer.param_groups:
				g['lr'] = g['lr'] * 0.1

		#auroc = evaluate(val_ld, selector, predictor, device, k)
		#print("Meta-val average AUROC: %.4f" % auroc)

		for i, data in enumerate(train_ld):
			optimizer.zero_grad()
			pool = data['pool'].numpy()
			pool_labels = data['pool_labels'].numpy()

			data['pool'] = data['pool'].to(device)
			logits = selector(data['pool'][:,:,ssidx:seidx])
			idx, log_prob = sample_once(logits, k) #(batch_size, k)

			selected = []
			selected_labels = []
			for p in range(len(idx)):
				selected.append(pool[p][idx[p]])
				selected_labels.append(pool_labels[p][idx[p]])

			selected = torch.Tensor(np.array(selected))
			selected_labels = torch.LongTensor(np.array(selected_labels))

			preds = predictor.forward_selected(selected[:,:,:512], selected_labels, data['query'][:,:,:512], data['query_labels'])
			preds = preds.squeeze(dim=2) #(batch_size, query_set_size)

			res = np.array([0]*data['query_labels'].shape[0]).astype(float) #(batch_size,)
			for p in range(data['query_labels'].shape[0]):
				res[p] = roc_auc_score(data['query_labels'][p,:],preds[p,:])
			main_reward = torch.Tensor(res)

			baseline_reward = random_baseline(data, predictor, k)
			reward = main_reward - baseline_reward #(batch_size,)
			reward = reward.to(device)

			final = -(reward * log_prob) #(batch_size,)
			final = torch.mean(final)
			final.backward()
			optimizer.step()

			report_reward += main_reward.mean()
			report_examples += 1

			if (i+1) % 5 == 0:
				print('epoch %d, iter %d, avg_reward %.3f, time_elapsed %.3f sec' % (epoch+1, i+1,
											           report_reward/report_examples,
												   time.time() - begin_time))
				report_reward = 0.0
				report_examples = 0
			if (i+1) % 30 == 0:
				auroc, _ = evaluate(val_ld, selector, predictor, device, k, return_avg=True)
				print("Meta-val average AUROC: %.4f" % auroc)

				if best_metric is None or auroc > best_metric: #new best network
					print("saving new best network!\n")
					best_metric = auroc
					path = os.path.join(save_path, "model_epoch%d_iter%d" % (epoch+1, i+1))
					torch.save({'epoch': epoch+1,
						    'model_state_dict': selector.state_dict(),
						    'optimizer_state_dict': optimizer.state_dict()},
						   path)

                                        
def load_data(positive_csv_path, normal_csv_path, num_tasks, num_workers=4, deterministic=True):
        """Return a dataloader given positive and normal X-ray data.
        Args:
            positive_csv_path (string): path to csv containing data for X-ray images that are positive 
                                        for abnormalities
            normal_csv_path (string): path to csv containing data for X-ray images that are labeled
                                      positive for "No Finding", meaning they are normal
            num_tasks (int): number of tasks to sample for this dataset 
            num_workers (int): number of worker threads to load the data
            deterministic (bool): whether the tasks in a dataset are sampled deterministically. If
                                  deterministic, the tasks used are the same across different epochs

        Returns:
            ld (torch.utils.data.DataLoader): dataloader for the given data
        """
        dset = RandomTaskDataset(positive_csv_path=positive_csv_path,
                                 normal_csv_path=normal_csv_path,
                                 unlabeled_pool_size=UNLABELED_POOL_SIZE,
                                 unlabeled_pos_frac=UNLABELED_POS_FRAC,
                                 query_set_size=QUERY_SET_SIZE,
                                 query_pos_frac=QUERY_POS_FRAC,
                                 conditions_used=NON_HOLD_OUT,
                                 num_tasks=num_tasks,
                                 deterministic=deterministic,
                                 use_asl=USE_ASL)
        ld = torch.utils.data.DataLoader(dataset=dset,
                                         batch_size=BATCH_SIZE,
                                         collate_fn=collate_fn,
                                         num_workers=num_workers)
        return ld
                                       

def train_helper(train_ld, val_ld, save_path):
        """Train MedSelect for several values of k.
        Args:
            train_ld (torch.utils.data.DataLoader): dataloader for train data 
            val_ld (torch.utils.data.DataLoader): dataloader for test data
            save_path (string): directory in which checkpoints will be saved
        """
        predictor = meanPred(mode = 'cosine')
        selector = LSTMSelector(input_size=(512 + USE_ASL*3))

        print("\n\nRunning for K=%d" % K)
        train_model(train_ld=train_ld,
                    val_ld=val_ld,
                    predictor=predictor,
                    selector=selector,
                    save_path=save_path,
                    num_epochs=NUM_EPOCHS,
                    lr=LEARNING_RATE,
                    k=K)
        
if __name__ == '__main__':
        prs = argparse.ArgumentParser(description='Train a MedSelect model.')
        prs.add_argument('--train_pos_csv', type=str, nargs='?', required=True,
                         help='Path to training set csv containing data for X-rays that are positive for abnormalities')
        prs.add_argument('--train_norm_csv', type=str, nargs='?', required=True,
                         help='Path to training set csv containing data for X-rays that are positive for No Finding')
        prs.add_argument('--val_pos_csv', type=str, nargs='?', required=True,
                         help='Path to val set csv containing data for X-rays that are positive for abnormalities')
        prs.add_argument('--val_norm_csv', type=str, nargs='?', required=True,
                         help='Path to val set csv containing data for X-rays that are positive for No Finding')
        prs.add_argument('--out', type=str, nargs='?', required=True,
                         help='Path to directory in which checkpoints will be saved')
        args = prs.parse_args()
        
        train_pos = args.train_pos_csv
        train_norm = args.train_norm_csv
        val_pos = args.val_pos_csv
        val_norm = args.val_norm_csv
        save_path = args.out
        
        train_ld = load_data(train_pos, train_norm, NUM_META_TRAIN_TASKS)
        val_ld = load_data(val_pos, val_norm, NUM_META_TEST_TASKS)
        train_helper(train_ld, val_ld, save_path)
