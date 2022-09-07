# GraphTorch

### Graph Neural Networks with Nested Tensors

## Summary

GraphTorch enables custom graph neural networks to be implemented within PyTorch, similar to [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html). However, GraphTorch utilises Torch's new [Nested Tensor](https://pytorch.org/docs/stable/nested.html) feature. This greatly simplifies the implementations of GNNs. Unlike in PyTorch Geometric, is not necessary to implement separate `message`, `aggregate`, and `edge_update` functions. Incoming edges at each node can simply be treated as an additional dimension in a PyTorch tensor.

This library contains a minimal example of how GraphTorch simplifies the implementation of a GNN compared to PyTorch. The file `gnn_geometric.py` contains the PyTorch Geometric implementation of a GNN, and `gnn.py` contains the equivalent GraphTorch implementation of the same GNN.
