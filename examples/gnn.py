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
		data_list = []
		for i in range(n):
			data_list.append(torch.cat([x1n[i].data, x1[i].unsqueeze(0).expand(x1n[i].data.shape[0], -1)], dim=-1))
		x1n_x1 = HetTensor(data=torch.cat(data_list, dim=0), idxs=x1n.idxs, dim_perm=x1n.dim_perm)
		a1 = self.atten(x1n_x1) # n x None x 1
		a2 = torch.softmax(a1, dim=1) # n x None x 1
		x2n = self.mul(x1n, a2) # n x None x 2*dim
		x3 = self.sum(x2n, dim=1) # n x dim
		x4 = self.phi(x3) # n x out_dim
		return x4


if __name__ == "__main__":
	x = torch.randn(4, 2)

	e = torch.tensor([[0, 0, 1, 2, 2, 2, 3],[0, 1, 1, 0, 1, 3, 2]]).T

	gnn = GNN(2, 4)

	y = gnn(x, e)

	breakpoint()