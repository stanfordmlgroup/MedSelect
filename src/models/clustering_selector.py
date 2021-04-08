import numpy as np
import torch
import torch.nn as nn
from sklearn_extra.cluster import KMedoids
from constants import *

class ClusteringSelector(nn.Module):
	"""Selector which does K-medoids on the pool set to produce num_to_select clusters
           and the cluster centers are chosen for labeling."""

	def __init__(self):
		"""Initialize the module """
		super(ClusteringSelector, self).__init__()

	def forward(self, data, num_to_select):
		"""Select X-rays from the pool for labeling.

		Args:
		    data (dict): a batch of data as returned by the RandomTaskDataset
		    num_to_select (int): number of X-rays to select from the pool for labeling

		Returns:
		    selection (LongTensor): tensor of shape (batch_size, pool_size) with 1 or 0 values,
                                            indicating which X-rays in the pool are selected.
		 """

		pool = data['pool'] #(batch_size, pool_size, embedding_size)
		selection = []
		for i in range(pool.shape[0]):
			curr_pool = pool[i].cpu().numpy()
			kmedoids = KMedoids(num_to_select, random_state=42, init='k-medoids++').fit(curr_pool)
			ctr_idx = kmedoids.medoid_indices_

			curr = np.zeros(curr_pool.shape[0])
			curr[ctr_idx] = 1
			selection.append(curr)

		selection = torch.LongTensor(selection)
		return selection


