# Initialization Functions in Needle

This document describes the initialization functions available in Needle for creating and initializing tensors with various patterns and distributions.

## Basic Initialization

### Zeros and Ones
```python
# Create tensor filled with zeros
zeros = needle.init.zeros(3, 4)  # Shape (3, 4)

# Create tensor filled with ones
ones = needle.init.ones(2, 3)    # Shape (2, 3)
```

### Constant Values
```python
# Create tensor filled with constant value
const = needle.init.full((2, 3), fill_value=5.0)
```

## Random Initialization

### Uniform Distribution
```python
# Random values from uniform distribution
uniform = needle.init.rand(3, 4)           # Default range [0, 1)
uniform = needle.init.uniform(3, 4, 
                            low=-1, 
                            high=1)         # Custom range
```

### Normal Distribution
```python
# Random values from normal distribution
normal = needle.init.randn(3, 4)           # Standard normal
normal = needle.init.normal(3, 4, 
                          mean=0, 
                          std=0.1)          # Custom parameters
```

## Neural Network Initializations

### Xavier/Glorot Initialization
```python
# Xavier uniform initialization
xavier_uniform = needle.init.xavier_uniform(in_dim, out_dim)

# Xavier normal initialization
xavier_normal = needle.init.xavier_normal(in_dim, out_dim)
```

### Kaiming/He Initialization
```python
# Kaiming uniform initialization
kaiming_uniform = needle.init.kaiming_uniform(in_dim, out_dim)

# Kaiming normal initialization
kaiming_normal = needle.init.kaiming_normal(in_dim, out_dim)
```

## Implementation Details

### Base Functions

1. **zeros**
   ```python
   def zeros(*shape, dtype="float32", device=None):
       """
       Creates a tensor filled with zeros
       
       Args:
           *shape: The shape of the tensor
           dtype: Data type of the tensor
           device: Device to create the tensor on
       """
   ```

2. **ones**
   ```python
   def ones(*shape, dtype="float32", device=None):
       """
       Creates a tensor filled with ones
       
       Args:
           *shape: The shape of the tensor
           dtype: Data type of the tensor
           device: Device to create the tensor on
       """
   ```

### Random Generators

1. **rand/uniform**
   ```python
   def uniform(low=0.0, high=1.0, *shape, dtype="float32", device=None):
       """
       Creates a tensor with random values from uniform distribution
       
       Args:
           low: Lower bound of the distribution
           high: Upper bound of the distribution
           *shape: The shape of the tensor
           dtype: Data type of the tensor
           device: Device to create the tensor on
       """
   ```

2. **randn/normal**
   ```python
   def normal(mean=0.0, std=1.0, *shape, dtype="float32", device=None):
       """
       Creates a tensor with random values from normal distribution
       
       Args:
           mean: Mean of the distribution
           std: Standard deviation of the distribution
           *shape: The shape of the tensor
           dtype: Data type of the tensor
           device: Device to create the tensor on
       """
   ```

## Weight Initialization Theory

### Xavier/Glorot Initialization
Designed to maintain variance across layers in networks using linear transformations:

- **Uniform**: \[ W \sim U\left(-\sqrt{\frac{6}{n_{in} + n_{out}}}, \sqrt{\frac{6}{n_{in} + n_{out}}}\right) \]
- **Normal**: \[ W \sim N\left(0, \sqrt{\frac{2}{n_{in} + n_{out}}}\right) \]

### Kaiming/He Initialization
Designed for networks using ReLU activation:

- **Uniform**: \[ W \sim U\left(-\sqrt{\frac{6}{n_{in}}}, \sqrt{\frac{6}{n_{in}}}\right) \]
- **Normal**: \[ W \sim N\left(0, \sqrt{\frac{2}{n_{in}}}\right) \]

## Best Practices

1. **Choosing Initialization**
   - Use Xavier for tanh/sigmoid activations
   - Use Kaiming for ReLU activations
   - Consider layer width when choosing parameters

2. **Numerical Stability**
   - Initialize biases to small values or zero
   - Scale initialization based on layer width
   - Consider gradient flow in deep networks

3. **Random Seeds**
   ```python
   # Set random seed for reproducibility
   needle.init.seed(42)
   ```

## Example Usage

```python
import needle as ndl

# Create a neural network layer
in_features = 784
out_features = 512

# Initialize weights with Xavier normal
weights = ndl.init.xavier_normal(in_features, out_features)

# Initialize biases with zeros
biases = ndl.init.zeros(out_features)

# Create layer output
output = input @ weights + biases
```

## Custom Initialization

To create a custom initialization:

```python
def custom_init(*shape, dtype="float32", device=None):
    """Custom initialization function"""
    data = # Your initialization logic here
    return needle.Tensor(data, dtype=dtype, device=device)
```