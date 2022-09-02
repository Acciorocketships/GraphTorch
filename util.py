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


def collect(input, index, dim_along=0, dim_select=1):
	'''
	input is the source tensor, of any shape
	index is a 1D tensor of shape input.shape[dim_along]
	dim_along is the dimension where each element you would like a different value
	dim_select is the dimension that elements in index are indexing

	For example, if you have input: (B, T, D) and index: (B,) where each element is in range(0, T),
	you can select the t-th element from each row in the batch dimension to produce an output of shape (B, D) with the following:
	output = collect(input, index, along_dim=0, dim_select=1)
	because we are indexing along dimension 0 (the B dim), and each element in index picks out an element in dimension 1 (the T dim).
	'''
	shape = list(input.shape)
	shape[dim_along] = index.shape[0]
	shape[dim_select] = 1
	unsqueeze_shape = [1] * len(shape)
	unsqueeze_shape[dim_along] = index.shape[0]
	index_unsqueezed = index.view(*unsqueeze_shape)
	index_expanded = index_unsqueezed.expand(*shape)
	out = torch.gather(input=input, index=index_expanded, dim=dim_select)
	return out.squeeze(dim_select)