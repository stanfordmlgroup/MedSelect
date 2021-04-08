import torch
import torch.nn as nn
import numpy as np

class TransformerSelector(nn.Module):
	"""Selector model based on a transformer"""

	def __init__(self, input_size=512, num_layers=2, nhead=8):
		"""Initialize the Selector module.

			Args:
			    input_size (int): dimension of the X-ray's embedding vector
			    num_layers (int): number of stacked transformer layers
			    nhead (int): number of attention heads
		 """
		super(TransformerSelector, self).__init__()
		self.input_size = input_size
		self.num_layers = num_layers
		self.nhead = nhead

		encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=nhead)
		self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
		self.linear = nn.Linear(in_features=input_size, out_features=1)

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
		out = self.transformer(pool) #(batch_size, pool_size, embedding_dim)
		out = self.linear(out) #(batch_size, pool_size, 1)
		return out.squeeze(dim=2)

if __name__ == '__main__':
	selector = TransformerSelector(input_size=512)
	dummy_pool = torch.rand(4, 1000, 512)
	out = selector(dummy_pool)
	print(out.shape)
