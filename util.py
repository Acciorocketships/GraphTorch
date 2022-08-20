import torch

def scatter_nested(input, idxi, idxj):
	idxi, perm = torch.sort(idxi)
	idxj = idxj[perm]
	num_nodes = input.shape[0]
	sizes = torch.zeros(num_nodes).int().scatter_(dim=0, index=idxi, src=torch.ones_like(idxi).int(), reduce='add')
	out_flat = torch.index_select(input=input, dim=0, index=idxj)
	out = torch.nested_tensor(torch.split(out_flat, sizes.tolist()))
	return out

def sum_nested(input, dim):
	dim = dim - 1 if dim > 0 else dim
	x = [xi.sum(dim=dim) for xi in input.unbind()]
	if all(map(lambda xi: xi.shape == x[0].shape, x)):
		return torch.stack(x, dim=0)
	else:
		return torch.nested_tensor(x)

def size_nested(input, dim):
	if dim == 0:
		return len(input._nested_tensor_size())
	dim = dim - 1 if dim > 0 else dim
	return torch.tensor([size[dim] for size in input._nested_tensor_size()])

def cat_nested(input1, input2, dim):
	if dim == 0:
		return torch.nested_tensor(input1.unbind() + input2.unbind())
	dim = dim - 1 if dim > 0 else dim
	if (input1.is_nested) and (not input2.is_nested):
		dim0_size = len(input1.unbind())
		input2 = input2.expand(dim0_size, *input2.shape[1:])
	if (input2.is_nested) and (not input1.is_nested):
		dim0_size = len(input2.unbind())
		input1 = input1.expand(dim0_size, *input1.shape[1:])
	needs_expansion = lambda x, y: (x.narrow(dim=dim, start=0, length=1).numel() < y.narrow(dim=dim, start=0, length=1).numel())
	expand_sizes = lambda y: list(y.shape[:dim]) + [-1] + list(y.shape[(dim+1 if dim >=0 else dim+1+len(y.shape)):])
	return torch.nested_tensor([
			torch.cat([
				x1i.expand(*expand_sizes(x2i)) if needs_expansion(x1i, x2i) else x1i,
				x2i.expand(*expand_sizes(x1i)) if needs_expansion(x2i, x1i) else x2i,
			],dim=dim)
		for (x1i, x2i) in zip(input1.unbind(), input2.unbind())])

def mul_nested(input1, input2):
	return torch.nested_tensor([x1i * x2i for (x1i, x2i) in zip(input1.unbind(), input2.unbind())])