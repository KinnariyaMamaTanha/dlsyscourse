# Operations

The operations module (`ops`) in Needle provides a collection of mathematical operations that can be performed on tensors. These operations form the building blocks for creating computational graphs with automatic differentiation support.

## Structure

The operations are organized in the `ops` directory:
- `__init__.py` - Imports all operations from submodules
- `ops_mathematic.py` - Contains mathematical operations like addition, multiplication, etc.

## Mathematical Operations

### Element-wise Operations

- **EWiseAdd / add**: Element-wise addition of two tensors
- **AddScalar / add_scalar**: Add a scalar value to a tensor
- **EWiseMul / multiply**: Element-wise multiplication of two tensors
- **MulScalar / mul_scalar**: Multiply a tensor by a scalar
- **EWisePow / power**: Element-wise power operation
- **PowerScalar / power_scalar**: Raise tensor elements to a scalar power
- **EWiseDiv / divide**: Element-wise division of two tensors
- **DivScalar / divide_scalar**: Divide a tensor by a scalar
- **Negate / negate**: Negate the elements of a tensor
- **Log / log**: Element-wise natural logarithm
- **Exp / exp**: Element-wise exponential function
- **ReLU / relu**: Rectified Linear Unit activation function

### Tensor Manipulation

- **Transpose / transpose**: Permute the dimensions of a tensor
- **Reshape / reshape**: Change the shape of a tensor without changing its data
- **BroadcastTo / broadcast_to**: Broadcast a tensor to a new shape
- **Summation / summation**: Sum tensor elements along specified axes
- **MatMul / matmul**: Matrix multiplication

## Implementation Details

Each operation is implemented as a subclass of `TensorOp` from the `autograd` module and provides two key methods:

1. **compute**: Implements the forward pass of the operation using NumPy arrays
2. **gradient**: Implements the backward pass (gradient computation) for use in automatic differentiation

Example (EWiseAdd):
```python
class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad
```

## Usage

Operations can be used either directly or via operator overloading on Tensor objects:

```python
import needle as ndl

# Direct usage
a = ndl.Tensor([1, 2, 3])
b = ndl.Tensor([4, 5, 6])
c = ndl.ops.add(a, b)  # [5, 7, 9]

# Operator overloading
d = a + b  # Same as above
e = a * 2  # Uses mul_scalar
f = a @ b  # Matrix multiplication
```

## Extension

To add a new operation:

1. Create a new class inheriting from `TensorOp`
2. Implement the `compute` method for forward pass
3. Implement the `gradient` method for backward pass
4. Add a helper function that creates and calls an instance of your operation class
5. (Optional) Add operator overloading in the `Tensor` class if applicable