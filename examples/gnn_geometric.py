import torch
from graphtorch import build_mlp
from torch_scatter.composite import scatter_softmax
from torch_scatter import scatter_sum
from torch_geometric.nn import MessagePassing as GeomMessagePassing


class GNN_Geometric(GeomMessagePassing):

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
		# edge_index: 2 x edges
		x1 = self.psi(x) # n x dim
		a2 = self.edge_updater(edge_index=edge_index, x=x1) # edges x 1
		x3 = self.propagate(edge_index=edge_index, x=x1, a=a2) # n x dim
		x4 = self.phi(x3) # n x out_dim
		return x4


	def message(self, x_j, a):
		x2n = x_j * a # edges x dim
		return x2n


	def aggregate(self, inputs, index, dim_size):
		x2n = inputs # edges x dim
		x3 = scatter_sum(src=x2n, index=index, dim=0, dim_size=dim_size) # n x dim
		return x3


	def edge_update(self, x_i, x_j, index, size_i):
		xi_xj = torch.cat([x_i, x_j], dim=-1)
		a1 = self.atten(xi_xj) # edges x 1
		a2 = scatter_softmax(src=a1, index=index, dim=0, dim_size=size_i) # edges x 1
		return a2


if __name__ == "__main__":
	x = torch.randn(4, 2)

	e = torch.tensor([[0, 0, 1, 2, 2, 2, 3],
					  [0, 1, 1, 0, 1, 3, 2]])

	gnn = GNN_Geometric(2, 4)

	y = gnn(x, e)

	breakpoint()