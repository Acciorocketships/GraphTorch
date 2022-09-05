import torch
from graphtorch.util import *

class MessagePassing(torch.nn.Module):

	def __init__(self):
		super().__init__()
		self.scatter = scatter_nested
		self.sum = sum_nested
		self.cat = cat_nested
		self.size = size_nested
		self.mul = mul_nested

	def propagate(self, x: torch.Tensor, edge_index: torch.Tensor, dim=0):
		if dim > 0:
			return torch.nested_tensor([self.propagate(xi, edge_index, dim=dim-1) for xi in x.unbind()])
		neighbours = scatter_nested(input=x, idxi=edge_index[0,:], idxj=edge_index[1,:])
		return neighbours


if __name__ == "__main__":
	x = torch.rand(4, 2)
	e = torch.tensor([[0, 0, 1, 2, 2, 2, 3],
					  [0, 1, 1, 0, 1, 3, 2]])
	y = scatter_nested(input=x, idxi=e[0,:], idxj=e[1,:])
	a = torch.ones(1,2).unsqueeze(0)
	b = cat_nested(y, a, dim=1)

