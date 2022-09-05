import torch.nn as nn
import numpy as np


def build_mlp(input_dim, output_dim, nlayers=1, midmult=1., batchnorm=False, layernorm=True, nonlinearity=nn.GELU):
	mlp_layers = layergen(input_dim=input_dim, output_dim=output_dim, nlayers=nlayers, midmult=midmult)
	mlp = MLP(layer_sizes=mlp_layers, batchnorm=batchnorm, layernorm=layernorm, nonlinearity=nonlinearity)
	return mlp


def layergen(input_dim, output_dim, nlayers=1, midmult=1.0):
	midlayersize = midmult * (input_dim + output_dim) // 2
	midlayersize = max(midlayersize, 1)
	nlayers += 2
	layers1 = np.around(
		np.logspace(np.log10(input_dim), np.log10(midlayersize), num=(nlayers) // 2)
	).astype(int)
	layers2 = np.around(
		np.logspace(
			np.log10(midlayersize), np.log10(output_dim), num=(nlayers + 1) // 2
		)
	).astype(int)[1:]
	return list(np.concatenate([layers1, layers2]))



class MLP(nn.Module):
	def __init__(
		self,
		layer_sizes,
		batchnorm=False,
		layernorm=False,
		nonlinearity=nn.ReLU,
	):
		super(MLP, self).__init__()
		self.batchnorm = batchnorm
		layers = []
		for i in range(len(layer_sizes) - 1):
			layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
			if i != len(layer_sizes) - 2:
				if batchnorm:
					layers.append(nn.BatchNorm(layer_sizes[i + 1]))
				if layernorm:
					layers.append(nn.LayerNorm(layer_sizes[i + 1]))
				layers.append(nonlinearity())
		self.net = nn.Sequential(*layers)

	def forward(self, X):
		return self.net(X)