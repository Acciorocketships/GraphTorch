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
		if len(shape_nested(edge_index)) > 2:
			return [self.propagate(x_i, edge_index_i, dim=dim-1) for (x_i, edge_index_i) in zip(x.unbind(), edge_index.unbind())]
		num_agents = size_nested(x, dim=0)
		neighbours = scatter_nested(input=x, idxi=edge_index[:,0], idxj=edge_index[:,1], size=num_agents)
		return neighbours


if __name__ == "__main__":
	x = torch.rand(4, 2)
	e = torch.tensor([[0, 0, 1, 2, 2, 2, 3],[0, 1, 1, 0, 1, 3, 2]]).T
	m = MessagePassing()
	y1 = m.propagate(x, e)
	y2 = m.propagate(torch.nested_tensor([x]), torch.nested_tensor([e]))
	breakpoint()

