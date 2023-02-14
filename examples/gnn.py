import torch
from hettensor import HetTensor
from graphtorch import build_mlp
from graphtorch import MessagePassing

class GNN(MessagePassing):

	def __init__(self, in_dim, out_dim):
		super().__init__()
		self.in_dim = in_dim
		self.out_dim = out_dim
		self.hidden_dim = (in_dim + out_dim) // 2
		self.psi = build_mlp(self.in_dim, self.hidden_dim, nlayers=3, layernorm=False)
		self.phi = build_mlp(self.hidden_dim, self.out_dim, nlayers=3, layernorm=False)
		self.atten = build_mlp(2 * self.hidden_dim, 1, nlayers=3, layernorm=False)

	def forward(self, x, edge_index):
		# x: n x in_dim
		# edge_index: edges x 2
		n = x.shape[0]
		x1 = self.psi(x) # n x dim
		x1n = self.propagate(x1, edge_index, dim=0) # n x None x dim
		x1n_x1 = x1n.apply(x1, lambda x, y: torch.cat([x, y.expand(x.shape[0], -1)], dim=-1), batch_dims=[0])
		a1 = x1n_x1.apply(None, self.atten) # n x None x 1
		a2 = a1.apply(None, lambda x: torch.softmax(x, dim=0), batch_dims=[0]) # n x None x 1
		x2n = x1n * a2 # n x None x 2*dim
		x3 = x2n.sum(dim=1) # n x dim
		x3t = x3.data
		x4 = self.phi(x3t) # n x out_dim
		return x4


if __name__ == "__main__":
	x = torch.randn(4, 2)

	e = torch.tensor([[0, 0, 1, 2, 2, 2, 3],[0, 1, 1, 0, 1, 3, 2]]).T

	gnn = GNN(2, 4)

	y = gnn(x, e)

	breakpoint()