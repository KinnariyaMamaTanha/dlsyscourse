# Automatic Differentiation in Needle

Needle implements automatic differentiation through a dynamic computation graph. This document explains the core components and how they work together.

## Core Components

### Value Class
The base class for all values in the computational graph.

```python
class Value:
    op: Optional[Op]        # The operator that produced this value
    inputs: List["Value"]   # Input values to the operator
    cached_data: NDArray    # Cached computed data
    requires_grad: bool     # Whether this value requires gradients
```

Key methods:
- `realize_cached_data()`: Computes and caches the value
- `is_leaf()`: Checks if this is a leaf node (no operator)
- `backward()`: Computes gradients through the computation graph

### Tensor Class
The main user-facing class representing multi-dimensional arrays.

```python
class Tensor(Value):
    grad: "Tensor"  # Gradient of this tensor
```

Key features:
- Supports standard mathematical operations (+, -, *, /, @)
- Provides shape manipulation (reshape, transpose)
- Enables automatic gradient computation
- Integrates with NumPy backend

### Operator (Op) Class
Base class for all operations in Needle.

```python
class Op:
    def compute(self, *args: Tuple[NDArray]) -> NDArray
    def gradient(self, out_grad: "Value", node: "Value") -> Union["Value", Tuple["Value"]]
```

Key responsibilities:
- Forward computation (`compute`)
- Gradient computation (`gradient`)
- Input/output handling

## How Automatic Differentiation Works

1. **Forward Pass**
   - Creates nodes in computation graph
   - Computes and caches values
   - Tracks dependencies between operations

2. **Backward Pass**
   - Starts from output node
   - Traverses graph in reverse topological order
   - Accumulates gradients using chain rule

3. **Gradient Computation**
   - Each operation defines its gradient computation
   - Gradients are accumulated at nodes
   - Leaf nodes store final gradients

## Example Usage

```python
# Create tensors
x = needle.Tensor([1, 2, 3], requires_grad=True)
y = needle.Tensor([4, 5, 6], requires_grad=True)

# Forward computation
z = x * y + y

# Backward pass
z.backward()

# Access gradients
print(x.grad)  # dy/dx
print(y.grad)  # dy/dy
```

## Implementation Details

### Topological Sort
The `find_topo_sort` function orders nodes for gradient computation:
```python
def find_topo_sort(node_list: List[Value]) -> List[Value]:
    """Returns nodes in reverse dependency order"""
```

### Gradient Computation
The `compute_gradient_of_variables` function handles gradient propagation:
```python
def compute_gradient_of_variables(output_tensor, out_grad):
    """Computes gradients of output w.r.t. each node"""
```

## Best Practices

1. **Memory Management**
   - Use `detach()` to create views without gradient tracking
   - Clear unnecessary intermediate values

2. **Gradient Computation**
   - Set `requires_grad=False` for constant tensors
   - Use `backward()` only on scalar outputs
   - Handle non-scalar outputs with appropriate gradients

3. **Performance**
   - Cache computed values when possible
   - Avoid unnecessary gradient computations
   - Use in-place operations when appropriate