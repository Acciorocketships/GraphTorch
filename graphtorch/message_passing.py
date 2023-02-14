from hettensor import HetTensor
from torch import Tensor
from typing import Union
from hettensor.hettensor import running_counts
from graphtorch.util import *

class MessagePassing(torch.nn.Module):

	def __init__(self):
		super().__init__()

	def propagate(self, x: Union[Tensor, HetTensor], edge_index: Union[Tensor, HetTensor], dim=0):
		if isinstance(edge_index, HetTensor):
			n_batch_dims = edge_index.dim() - 2
			batch_dims = list(range(n_batch_dims))
			neighbours = x.scatter(edge_index, batch_dims=batch_dims, index_dim=dim, src_sink=True)
			return neighbours
		elif isinstance(edge_index, torch.Tensor):
			idxi = edge_index[:,0]
			idxj = edge_index[:,1]
			data = torch.index_select(input=x, dim=0, index=idxi)
			idx_el = running_counts(idxj)
			idxs = torch.cat([idxj.unsqueeze(0), idx_el.unsqueeze(0)], dim=0)
			dim_perm = list(range(data.dim() + 1))
			neighbours = HetTensor(data=data, idxs=idxs, dim_perm=dim_perm)
			return neighbours


if __name__ == "__main__":
	x1 = torch.rand(4, 2)
	x2 = torch.rand(3, 2)
	e1 = torch.tensor([[0, 0, 1, 2, 2, 2, 3],[0, 1, 1, 0, 1, 3, 2]]).T
	e2 = torch.tensor([[0, 1, 1, 2], [2, 1, 3, 2]]).T
	x = HetTensor([x1, x2])
	e = HetTensor([e1, e2])
	# y = x.scatter(idxs=e, batch_dims=[0], index_dim=1)
	m = MessagePassing()
	y = m.propagate(x, e)
	breakpoint()
