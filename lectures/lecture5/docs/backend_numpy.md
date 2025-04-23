# Backend NumPy

The `backend_numpy.py` module provides device implementations for the Needle framework using NumPy as the underlying array library. This module defines how tensor operations are performed on CPU devices.

## Key Components

### 1. Device Classes

- **Device**: Base class for all device implementations
- **CPUDevice**: Implementation for CPU-based computations
  - Provides methods for array creation and manipulation (e.g., `zeros`, `ones`, `randn`, `rand`)
  - Implements operations for creating arrays with specific patterns (`one_hot`, `empty`, `full`)
  - Encapsulates all NumPy operations needed for tensor computations

### 2. Device Management Functions

- **cpu()**: Returns a CPUDevice instance
- **default_device()**: Returns the default device (currently CPU)
- **all_devices()**: Returns a list of all available devices in the system

## Device Operations

The following operations are supported by the CPUDevice:

- **zeros**: Create an array filled with zeros of specified shape and dtype
- **ones**: Create an array filled with ones of specified shape and dtype
- **randn**: Generate random numbers from standard normal distribution
- **rand**: Generate random numbers from uniform distribution
- **one_hot**: Create a one-hot encoded array
- **empty**: Create an uninitialized array of specified shape
- **full**: Create an array filled with a specified value

## Usage

The backend is typically not directly accessed by end users but is used internally by the Tensor class. However, you can explicitly specify devices when creating tensors:

```python
import needle as ndl

# Create a CPU device
cpu_device = ndl.cpu()

# Create a tensor on the CPU device
x = ndl.Tensor([1, 2, 3], device=cpu_device)
```

## Extension Points

The backend system is designed to be extensible. Future implementations could include:

- GPU-based backends (e.g., CUDA)
- Other array libraries as backends (e.g., JAX, PyTorch, TensorFlow)
- Specialized hardware backends (e.g., TPU, custom accelerators)

To add a new backend, implement a subclass of `Device` that provides all the required array operations.