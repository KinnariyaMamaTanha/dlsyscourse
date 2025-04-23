# Initialization

The initialization module (`init`) provides functions for creating and initializing tensors with specific patterns or random values. These are essential for setting up neural network parameters with proper initial values.

## Structure

The initialization module is organized in the `init` directory:
- `__init__.py` - Imports all initialization functions from submodules
- `init_basic.py` - Contains basic initialization functions

## Initialization Functions

### Random Initializations

- **rand**: Generate tensors with random values from a uniform distribution between `low` and `high`
  ```python
  rand(*shape, low=0.0, high=1.0, device=None, dtype="float32", requires_grad=False)
  ```

- **randn**: Generate tensors with random values from a normal distribution with `mean` and `std`
  ```python
  randn(*shape, mean=0.0, std=1.0, device=None, dtype="float32", requires_grad=False)
  ```

- **randb**: Generate binary random tensors (True/False) with probability `p` of being True
  ```python
  randb(*shape, p=0.5, device=None, dtype="bool", requires_grad=False)
  ```

### Constant Initializations

- **constant**: Generate tensors filled with a constant value `c`
  ```python
  constant(*shape, c=1.0, device=None, dtype="float32", requires_grad=False)
  ```

- **ones**: Generate tensors filled with ones (shortcut for `constant` with `c=1.0`)
  ```python
  ones(*shape, device=None, dtype="float32", requires_grad=False)
  ```

- **zeros**: Generate tensors filled with zeros (shortcut for `constant` with `c=0.0`)
  ```python
  zeros(*shape, device=None, dtype="float32", requires_grad=False)
  ```

### Special Initializations

- **one_hot**: Generate one-hot encoding tensor
  ```python
  one_hot(n, i, device=None, dtype="float32", requires_grad=False)
  ```

### Clone-based Initializations

- **zeros_like**: Create a tensor of zeros with the same shape, dtype, and device as the input tensor
  ```python
  zeros_like(array, *, device=None, requires_grad=False)
  ```

- **ones_like**: Create a tensor of ones with the same shape, dtype, and device as the input tensor
  ```python
  ones_like(array, *, device=None, requires_grad=False)
  ```

## Common Parameters

Most initialization functions share common parameters:

- **shape**: The shape of the tensor to be created (passed as positional arguments)
- **device**: The device where the tensor will be stored (CPU by default)
- **dtype**: The data type of the tensor ("float32" by default)
- **requires_grad**: Whether the tensor requires gradient computation (False by default)

## Usage Examples

```python
import needle as ndl

# Create a 3x3 tensor filled with random values from a normal distribution
weights = ndl.init.randn(3, 3, std=0.01, requires_grad=True)

# Create a vector of zeros
bias = ndl.init.zeros(3, requires_grad=True)

# Create a batch of one-hot encoded vectors
labels = ndl.init.one_hot(10, ndl.Tensor([0, 3, 9]))

# Create a tensor with same shape as another
weights_like = ndl.init.ones_like(weights)
```

## Implementation Details

All initialization functions use the device API (as defined in `backend_numpy.py`) to create the array with the specified pattern, then wrap it in a `Tensor` object. This ensures that tensors can be created on any supported device with the same API.