# GraphTorch

### Graph Neural Networks with Nested Tensors

Similar to [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html), GraphTorch custom graph neural networks to be implemented within PyTorch. However, GraphTorch utilises Torch's new [Nested Tensor](https://pytorch.org/docs/stable/nested.html) feature. This greatly simplifies the implementations of GNNs. Unlike in PyTorch Geometric, is not necessary to implement separate `message`, `aggregate`, and `edge_update` functions. Incoming edges at each node can simply be treated as an additional dimension in a PyTorch tensor.

This library contains a minimal example of how GraphTorch simplifies the implementation of a GNN compared to PyTorch. The file `gnn_geometric.py` contains the PyTorch Geometric implementation of a GNN, and `gnn.py` contains the equivalent GraphTorch implementation of the same GNN.

Currently, Nested Tensor is still under development, so it does not natively support some operations like sum, mul, cat, and advanced indexing. As a temporary solution, GraphTorch's MessagePassing base class contains python implementations for these operations. Hopefully, support for Nested Tensor will increase in the future, and these temporary workarounds can be replaced.

When more advanced indexing becomes available (for example, using `torch.gather` with a Nested Tensor as indices), it will be possible to further extend GraphTorch. Currently, it handles batches in the same way as PyTorch Geometric: by combining all graphs into a a single graph. However, since Nested Tensor can contain Tensors of variable sizes, in the future it will be possible to add a batch dimension, so the input is of shape `Batch x Num_Nodes x Dim` and the edges are of shape `Batch x Num_Edges x 2`.
