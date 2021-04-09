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

	ssidx = 0 if USE_IMG else 512  # Start Index of Pool for Selector
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

if __name__ == '__main__':
	train_dset = RandomTaskDataset(positive_csv_path='/deep/group/activelearn/data/level1/meta_train/positives.csv',
				      normal_csv_path='/deep/group/activelearn/data/level1/meta_train/no_findings.csv',
				      unlabeled_pool_size=UNLABELED_POOL_SIZE,
				      unlabeled_pos_frac=UNLABELED_POS_FRAC,
				      query_set_size=QUERY_SET_SIZE,
				      query_pos_frac=QUERY_POS_FRAC,
				      conditions_used=NON_HOLD_OUT,
				      num_tasks=NUM_META_TRAIN_TASKS,
				      deterministic=True)
	train_ld = torch.utils.data.DataLoader(dataset=train_dset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=4)

	val_dset = RandomTaskDataset(positive_csv_path='/deep/group/activelearn/data/level1/meta_val/positives.csv',
			            normal_csv_path='/deep/group/activelearn/data/level1/meta_val/no_findings.csv',
				    unlabeled_pool_size=UNLABELED_POOL_SIZE,
                                    unlabeled_pos_frac=UNLABELED_POS_FRAC,
                                    query_set_size=QUERY_SET_SIZE,
                                    query_pos_frac=QUERY_POS_FRAC,
                                    conditions_used=NON_HOLD_OUT,
                                    num_tasks=NUM_META_TEST_TASKS,
                                    deterministic=True)
	val_ld = torch.utils.data.DataLoader(dataset=val_dset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=4)

	predictor = meanPred(mode = 'cosine')
	selector = LSTMSelector(input_size=(USE_IMG*512 + USE_ASL*3))

	for k in K:
		print(f"\n\nRunning for K={k}")
		train_model(train_ld=train_ld,
			    val_ld=val_ld,
			    predictor=predictor,
			    selector=selector,
			    save_path='/deep/u/akshaysm/ckpt',
			    num_epochs=NUM_EPOCHS,
			    lr=LEARNING_RATE,
			    k=k)
