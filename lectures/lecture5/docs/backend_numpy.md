# NumPy Backend

The NumPy backend provides the fundamental array operations for Needle. It implements device management and array operations using NumPy as the underlying computation engine.

## Device Management

### Device Class
```python
class Device:
    """
    Represents devices like CPU/GPU where tensors can be allocated
    """
    def __repr__(self):
        return self.device_name
    
    def __eq__(self, other):
        return self.device_name == other.device_name
```

### Available Devices

- **CPU Device**: Default device for computation
```python
class CPUDevice(Device):
    device_name = "cpu"
    
cpu = CPUDevice  # Default device
default_device = cpu()
```

## Array Interface

The backend provides a consistent interface for array operations that mirrors NumPy's functionality while allowing for future extensions to other backends.

### Key Features

1. **Array Creation**
   - From NumPy arrays
   - From Python lists/tuples
   - Random initialization

2. **Array Operations**
   - Element-wise operations
   - Matrix operations
   - Shape manipulation
   - Type conversion

### Core Functions

```python
def array(a, dtype=None, device=None):
    """Create an array on the specified device"""
    
def empty(shape, dtype=float32, device=None):
    """Create an uninitialized array"""
    
def full(shape, fill_value, dtype=float32, device=None):
    """Create an array filled with a scalar value"""
```

## Data Types

Supported NumPy data types:
- `float32` (default)
- `float64`
- `int32`
- `int64`

## Implementation Details

### Array Conversion
```python
def from_numpy(a, device=None):
    """Convert NumPy array to device array"""
    
def to_numpy(a):
    """Convert device array to NumPy array"""
```

### Memory Management
- Arrays are allocated on CPU memory
- Memory is managed by NumPy's memory allocator
- Explicit memory deallocation not required

## Best Practices

1. **Device Selection**
   ```python
   # Create array on CPU
   x = needle.array([1, 2, 3], device=cpu())
   ```

2. **Data Type Management**
   ```python
   # Specify data type explicitly
   x = needle.array([1, 2, 3], dtype="float32")
   ```

3. **Memory Efficiency**
   - Reuse arrays when possible
   - Use in-place operations
   - Clear references to unused arrays

4. **Performance Tips**
   - Batch operations when possible
   - Use appropriate data types
   - Minimize data transfers

## Example Usage

```python
import needle as ndl
from needle import backend_numpy as np

# Create array on CPU
x = ndl.array([1, 2, 3], device=cpu())

# Perform operations
y = x * 2
z = ndl.full(y.shape, 3.0)
result = y + z

# Convert back to NumPy if needed
numpy_result = result.numpy()
```

## Future Extensions

The backend system is designed to be extensible. Future versions may include:
- GPU support
- Other array libraries (CuPy, JAX, etc.)
- Custom device implementations