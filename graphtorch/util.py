import torch


def scatter_nested(input, idxi, idxj, size=None):
	'''
	:param input: The source tensor, which is either a normal tensor or nested_tensor
	:param idxi: A list of idxs in the range(input.shape[0]), denoting the idxs of the sink nodes for a list of edges
	:param idxj: A list of idxs in the range(input.shape[0]), denoting the idxs of the src nodes for a list of edges
	:return: A nested_tensor where out[i] contains stack([input[j] for all j where there exists an edge (j,i) from node i to node j])

	This functions as a simple way to create a nested tensor, where you simply specify which elements go in which column with idxi, idxj.
	For example, if you have a sparse matrix of shape (N x N x D) and an edge list [idxi, idxj] of shape (E x 2) specifying
	the nonzero elements, then calling scatter_nested will remove all of the zero elements.
	The resulting nested_tensor will be of shape (N x None x D), where out[i] will be the list of all nonzero elements
	in row i of the input (input[i]).
	'''
	idxi, perm = torch.sort(idxi, stable=True)
	idxj = idxj[perm]
	num_nodes = size if (size is not None) else (input.shape[0] if (not input.is_nested) else size_nested(input, dim=0))
	if input.is_nested:
		nested_shape = shape_nested(input)
		nested_dim = nested_shape.index(None) if (None in nested_shape) else 1
		out_list = [[] for _ in range(num_nodes)]
		for i, j in zip(idxi, idxj):
			out_list[i].append(input[j])
		out_list_cat = list(map(lambda nodei: torch.cat(nodei, dim=nested_dim-1), out_list))
		out = torch.nested_tensor(out_list_cat)
	else:
		sizes = torch.zeros(num_nodes).int().scatter_(dim=0, index=idxi, src=torch.ones_like(idxi).int(), reduce='add')
		out_flat = torch.index_select(input=input, dim=0, index=idxj)
		out = torch.nested_tensor(torch.split(out_flat, sizes.tolist()))
	return out


def apply_nested(func, input, *args):
	'''
	:param func:
	:param input:
	:param args:
	:return:

	Applies a function to each normal tensor within a nested_tensor. If more inputs are provided, then those are indexed
	and passed to the function too. That is, if you call apply_nested(f, x, n) where x is a nested_tensor and n is a tensor
	of sizes of the ragged dimension of x, then f(x[i], n[i]) will be called for each row of x, and the output will combine
	the results of f into a nested_tensor.

	This is particularly useful after calling scatter_nested with a nested_tensor as an input. Since nested_tensors cannot
	be stacked, the variable size dimensions must be combined into a single dimension in scatter_nested. However, apply_nested
	can be used to recover the original hidden dimension and apply transformations to it.
	'''
	# index_args = lambda args_list, i: [a[i] for a in args_list]
	def index_args(args_list, i):
		return [a[i] for a in args_list]
	return torch.nested_tensor([func(x, *index_args(args, i)) for i, x in enumerate(input.unbind())])


def sum_nested(input, dim):
	dim = dim - 1 if dim > 0 else dim
	x = [xi.sum(dim=dim) for xi in input.unbind()]
	if all(map(lambda xi: xi.shape == x[0].shape, x)):
		return torch.stack(x, dim=0)
	else:
		return torch.nested_tensor(x)


def size_nested(input, dim):
	if dim == 0:
		size = torch.tensor(len(input._nested_tensor_size()))
	else:
		dim = dim - 1 if dim > 0 else dim
		size = torch.tensor([size[dim] for size in input._nested_tensor_size()])
		if torch.all(size==size[0]):
			size = size[0]
	return size


def shape_nested(input):
	if input.is_nested:
		shape = []
		num_dims = len(input[0].shape) + 1
		for dim in range(num_dims):
			dim_size = size_nested(input, dim=dim)
			if len(dim_size.shape) > 0:
				dim_size = None
			else:
				dim_size = dim_size.item()
			shape.append(dim_size)
		return tuple(shape)
	else:
		return tuple(input.shape)


def cat_nested(input1, input2, dim):
	'''
	:param input1: First element to concatenate (tensor or nested_tensor)
	:param input2: Second element to concatenate (tensor or nested_tensor)
	:param dim: Dimension along which to concatenate
	:return: The concatenated nested_tensor

	input1 and input2 must have the same number of dimensions, and the sizes in all dimensions except dim must either match
	or be 1. If the dimension in one of the tensors is 1, then it is broadcast to the shape of the other tensor in that dimension.
	This functionality is important because it is non-trivial to broadcast manually with nested_tensors. For example, if
	we wish to concatenate input1: (B x 1 x D1) to input2: (B x None x D2) in dimension -1, producing an output: (B x None x (D1+D2)),
	then it would be nice if input1 were automatically broadcast to each element in the None dimension so that we do not first
	need to construct a nested_tensor.
	'''
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


def add_nested(input1, input2):
	return torch.nested_tensor([x1i + x2i for (x1i, x2i) in zip(input1.unbind(), input2.unbind())])


def nested_to_batch(nested, return_sizes=False):
	tensor_list = nested.unbind()
	flat = torch.cat(tensor_list, dim=0)
	sizes = size_nested(nested, dim=1)
	if return_sizes:
		return flat, sizes
	else:
		batch = torch.arange(len(tensor_list)).repeat_interleave(sizes)
		return flat, batch


def select_index(arr, dim, idx):
	idx_list = [slice(None)] * len(arr.shape)
	idx_list[dim] = idx
	return arr.__getitem__(idx_list)


def index_with_nested(src, index, dim=0):
	return torch.nested_tensor([select_index(src=src, idx=idxi.long(), dim=dim) for idxi in index.unbind()])


def permute_nested(nt, perm):
	if perm.is_nested:
		return torch.nested_tensor([xi[permi] for xi, permi in zip(nt.unbind(), perm.unbind())])
	else:
		dim0_size = size_nested(nt, dim=0)
		x_flat, sizes = nested_to_batch(nt, return_sizes=True)
		batch = torch.arange(dim0_size).repeat_interleave(sizes)
		x_perm = x_flat[perm]
		batch = batch[perm]
		return create_nested_batch(x_perm, batch, dim_size=dim0_size)


def create_nested(x, sizes):
	return torch.nested_tensor(torch.split(x, sizes.tolist()))


def create_nested_batch(x, batch, dim_size=None):
	'''
	Creates a nested tensor from a (n x d) tensor and a (n) vector specifying the batch of each element in that tensor
	:param x: a (n x d) tensor
	:param batch: a (n) vector specifying the batch along axis 0 of x
	:param dim_size: if given, it specifies b in the output shape (b x None x d). otherwise, b = max(batch)+1
	:return: a (b x None x d) tensor
	'''
	if dim_size is None:
		dim_size = torch.max(batch)+1
	sizes = torch.zeros(dim_size).scatter_(dim=0, index=batch, src=torch.ones(batch.shape[0]), reduce='add').int()
	return create_nested(x, sizes)


def truncate_nested(nt, sizes, dim=1):
	'''
	Sets the sizes of the variable size dimension in a nested tensor. The tensor is truncated when larger than the
	given size, and torch.nan is added when the tensor is smaller than the given size.
	:param nt: the source nested tensor
	:param sizes: a vector of sizes
	:param dim: the dim of nt to truncate
	:return: the updated version of nt
	'''
	dim = dim-1
	size_update = lambda size, dimsize: tuple(list(size)[:dim] + [dimsize] + list(size)[dim+1:])
	return torch.nested_tensor([
			torch.cat([
				torch.narrow(xi, dim=dim, start=0, length=min(size, xi.shape[dim])),
				torch.full(size=size_update(xi.shape, max(size-xi.shape[dim], 0)), fill_value=torch.nan)
			], dim=dim)
		for (xi, size) in zip(nt.unbind(), sizes)])


def collect(input, index, dim_along=0, dim_select=1):
	'''
	:param input: the source tensor, of any shape
	:param index: tensor of shape input.shape[dim_along], specifying the indices in dim_select to collect
	:param dim_along: the dimension where each element you would like a different value
	:param dim_select: the dimension that elements in index are indexing
	:return: an indexed version of the source tensor. it is missing the dimension dim_select, but otherwise it has the same shape

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