# Autograd

This module implements the core data structures and logic for the Needle automatic differentiation (autograd) framework. It provides the computational graph abstraction, operator definitions, and the mechanisms for forward and backward passes (gradient computation). The design is inspired by modern deep learning frameworks, supporting both eager and lazy evaluation.

## Key Components

### 1. Global Variables

- **LAZY_MODE**: Controls whether computation is performed eagerly or lazily.
- **TENSOR_COUNTER**: Tracks the number of active tensor objects (for debugging/memory management).

### 2. NDArray and Backend

- **NDArray**: Alias for `numpy.ndarray`. The backend can be swapped in future assignments.
- **array_api**: The array API used for computation, currently set to NumPy.

### 3. Operator Classes

- **Op**: Base class for all operators. Defines the interface for:
  - `__call__`: Operator invocation.
  - `compute`: Forward computation (to be implemented by subclasses).
  - `gradient`: Computes the partial adjoint (gradient) for each input.
  - `gradient_as_tuple`: Convenience method to always return a tuple from gradient call

- **TensorOp**: Subclass of `Op` for operators that output a single `Tensor`.
- **TensorTupleOp**: Subclass of `Op` for operators that output a `TensorTuple`.

### 4. Value Classes

- **Value**: Base class for nodes in the computational graph.
  - Tracks the operator (`op`), input nodes (`inputs`), cached data, and whether gradients are required.
  - Provides methods for realizing cached data, checking if a node is a leaf, and initialization.
  - Class methods for creating constant nodes and nodes from operators.

- **Tensor**: Subclass of `Value` representing a tensor in the graph.
  - Handles device, dtype, and data management.
  - Operator overloads for arithmetic and matrix operations.
  - Methods for backward pass (`backward`), detaching from the graph, and conversion to NumPy.
  - Properties for shape, dtype, and device.

- **TensorTuple**: Subclass of `Value` representing a tuple of tensors.
  - Supports tuple operations, indexing, and detachment.

### 5. Gradient Computation

- **compute_gradient_of_variables**: Given an output tensor and its gradient, traverses the computational graph in reverse topological order to compute and store gradients for all variables involved.

### 6. Graph Traversal

- **find_topo_sort**: Returns a topological ordering of nodes for a given list of output nodes using post-order DFS.
- **topo_sort_dfs**: Helper function for DFS traversal.

### 7. Helper Methods

- **sum_node_list**: Custom sum function to efficiently sum a list of nodes without creating redundant computation nodes.

## Usage

- **Tensor Creation**: Use `Tensor` to create new tensors, specifying data, device, dtype, and whether gradients are required.
- **Computation**: Use arithmetic operators or Needle ops to build computation graphs.
- **Backward Pass**: Call `.backward()` on a tensor to compute gradients with respect to all variables in the graph.
- **Lazy/Eager Mode**: Control evaluation strategy via `LAZY_MODE`.

## Extension Points

- Implementations for `compute_gradient_of_variables`, `find_topo_sort`, and `topo_sort_dfs` are required for full backward functionality.
- Operator subclasses (in `needle.ops`) must implement `compute` and `gradient` for each operation.

## Example

```python
import needle as ndl

x = ndl.Tensor([1, 2, 3], dtype="float32", requires_grad=True)
y = x + 1
z = y * 2
z.backward()
print(x.grad)  # Should print the gradient of z with respect to x
```