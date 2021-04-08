import torch
import torch.nn as nn
import numpy as np

class LSTMSelector(nn.Module):
	"""Selector model based on a BiLSTM"""

	def __init__(self, input_size, hidden_size=256, num_layers=1):
		"""Initialize the Selector module.

			Args:
			    input_size (int): dimension of the X-ray's embedding vector
			    hidden_size (int): number of hidden units in the BiLSTM
			    num_layers (int): number of stacked BiLSTM layers
		 """
		super(LSTMSelector, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.LSTM = nn.LSTM(input_size=input_size,
				    hidden_size=hidden_size,
				    num_layers=num_layers,
				    batch_first=True,
			    	    bidirectional=True)
		self.linear = nn.Linear(in_features=2*hidden_size, out_features=1)

	def forward(self, pool):
		"""Perform the forward pass through the model and return logits.

			Args:
			    pool (Tensor): A tensor of shape (batch_size, pool_size, input_size) which is the
                                           contains the unlabeled pool of X-rays for each task in the batch

			Returns:
			    out (Tensor): Tensor of shape (batch_size, pool_size) which is the output
                                          of the selector model. The outputs is raw logits, not a probability
                                          distribution
		"""
		out = self.LSTM(pool)[0]  #(batch_size, pool_size, 2*hidden_size)
		out = self.linear(out)    #(batch_size, pool_size, 1)
		return out.squeeze(dim=2)

if __name__ == '__main__':
	selector = LSTMSelector(input_size=512, hidden_size=64, num_layers=1)
	dummy_pool = torch.rand(4, 1000, 512)
	out = selector(dummy_pool)
	print(out.shape)
