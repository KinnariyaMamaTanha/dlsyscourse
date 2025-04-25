# Operations in Needle

This document describes the mathematical operations available in Needle and their implementations.

## Core Operation Types

### Element-wise Operations

1. **Basic Arithmetic**
   ```python
   # Addition
   z = x + y  # EWiseAdd
   z = x + 5  # AddScalar
   
   # Multiplication
   z = x * y  # EWiseMul
   z = x * 2  # MulScalar
   
   # Division
   z = x / y  # EWiseDiv
   z = x / 3  # DivScalar
   
   # Power
   z = x ** y  # EWisePow
   z = x ** 2  # PowerScalar
   ```

2. **Activation Functions**
   ```python
   # ReLU
   z = needle.ops.relu(x)
   
   # Sigmoid
   z = needle.ops.sigmoid(x)
   
   # Tanh
   z = needle.ops.tanh(x)
   ```

### Matrix Operations

1. **Matrix Multiplication**
   ```python
   # Using @ operator
   z = x @ y
   
   # Using matmul function
   z = needle.ops.matmul(x, y)
   ```

2. **Shape Operations**
   ```python
   # Reshape
   z = x.reshape((2, 3))
   
   # Transpose
   z = x.transpose()
   z = x.transpose((1, 0))  # With axes specified
   ```

### Reduction Operations

```python
# Sum
z = x.sum()           # Sum all elements
z = x.sum(axis=0)     # Sum along axis 0
z = x.sum(axis=(0,1)) # Sum along multiple axes

# Mean
z = needle.ops.mean(x)
z = needle.ops.mean(x, axis=0)
```

## Implementation Details

### Operation Base Class
```python
class Op:
    def compute(self, *args):
        """Forward computation"""
        pass
        
    def gradient(self, out_grad, node):
        """Gradient computation"""
        pass
```

### Forward Pass
Each operation implements the `compute` method:
```python
def compute(self, *args):
    """
    Args:
        *args: Input NDArrays
    Returns:
        NDArray: Result of the operation
    """
```

### Backward Pass
Each operation implements the `gradient` method:
```python
def gradient(self, out_grad, node):
    """
    Args:
        out_grad: Gradient from output
        node: The node this operation created
    Returns:
        Gradient(s) for input(s)
    """
```

## Common Operations Reference

### Mathematical Operations

1. **Addition (EWiseAdd)**
   - Forward: \( f(x,y) = x + y \)
   - Gradient: \( \frac{\partial f}{\partial x} = 1, \frac{\partial f}{\partial y} = 1 \)

2. **Multiplication (EWiseMul)**
   - Forward: \( f(x,y) = x * y \)
   - Gradient: \( \frac{\partial f}{\partial x} = y, \frac{\partial f}{\partial y} = x \)

3. **Matrix Multiplication (MatMul)**
   - Forward: \( f(X,Y) = XY \)
   - Gradient: 
     - \( \frac{\partial f}{\partial X} = \frac{\partial L}{\partial f}Y^T \)
     - \( \frac{\partial f}{\partial Y} = X^T\frac{\partial L}{\partial f} \)

### Shape Operations

1. **Reshape**
   - Forward: Reshapes input array to target shape
   - Gradient: Reshapes gradient back to input shape

2. **Transpose**
   - Forward: Permutes dimensions according to axes
   - Gradient: Inverse permutation of gradient

### Broadcasting Operations

1. **BroadcastTo**
   - Forward: Broadcasts input to larger shape
   - Gradient: Reduces gradient along broadcast axes

## Example Usage

```python
import needle as ndl

# Create input tensors
x = ndl.Tensor([[1, 2], [3, 4]], requires_grad=True)
y = ndl.Tensor([[5, 6], [7, 8]], requires_grad=True)

# Forward pass
z = (x @ y).sum()

# Backward pass
z.backward()

# Access gradients
print(x.grad)  # Gradient of z with respect to x
print(y.grad)  # Gradient of z with respect to y
```

## Best Practices

1. **Memory Efficiency**
   - Use in-place operations when possible
   - Clear intermediate results when not needed

2. **Numerical Stability**
   - Use stable implementations of operations
   - Handle edge cases in activation functions
   - Consider using log-space for certain operations

3. **Performance**
   - Batch operations when possible
   - Use appropriate data types
   - Minimize memory allocations

## Custom Operations

To implement a custom operation:

1. Create a new class inheriting from `TensorOp`
2. Implement `compute` method for forward pass
3. Implement `gradient` method for backward pass

Example:
```python
class CustomOp(TensorOp):
    def compute(self, x):
        return x * x
        
    def gradient(self, out_grad, node):
        x = node.inputs[0]
        return 2 * x * out_grad
```