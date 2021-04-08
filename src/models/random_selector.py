import numpy as np
import torch
import torch.nn as nn
#from constants import *

class RandomSelector(nn.Module):
	"""Selector which randomly selects X-rays from unlabeled pool."""

	def __init__(self):
		"""Initialize the module """
		super(RandomSelector, self).__init__()

	def forward(self, data, num_to_select):
		"""Select X-rays from the pool for labeling.

		Args:
		    data (dict): a batch of data as returned by the RandomTaskDataset
		    num_to_select (int): number of X-rays to select from the pool for labeling

		Returns:
		    selection (LongTensor): tensor of shape (batch_size, pool_size) with 1 or 0 values,
                                            indicating which X-rays in the pool are selected.
		 """
		np.random.seed(42)
		pool = data['pool'] #(batch_size, pool_size, embedding_size)
		selection = []
		for i in range(pool.shape[0]):
			curr_pool = pool[i].cpu().numpy()
			idx = np.random.choice(curr_pool.shape[0], size=num_to_select, replace=False)
			curr = np.zeros(curr_pool.shape[0])
			curr[idx] = 1
			selection.append(curr)

		selection = torch.LongTensor(selection)
		return selection

if __name__ == '__main__':
	rs = RandomSelector()
	inp = torch.rand(8, 100, 64)
	data = {'pool': inp}
	ret = rs(data, 20)
	print(ret)
