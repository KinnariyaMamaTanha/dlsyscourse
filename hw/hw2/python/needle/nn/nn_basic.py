"""The module."""

from math import log
from typing import List, Callable, Any

from ..autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: str = None,
        dtype: str = "float32",
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        weight = init.kaiming_uniform(
            in_features, out_features, device=device, dtype=dtype
        )
        self.weight = Parameter(weight)
        if bias:
            bias = init.kaiming_uniform(
                out_features, 1, device=device, dtype=dtype
            ).reshape((1, out_features))
            self.bias = Parameter(bias)
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return X @ self.weight + (self.bias if self.bias is not None else 0)
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        return ops.reshape(X, (X.shape[0], -1))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        bs = logits.shape[0]
        num_classes = logits.shape[1]
        logsumexp = ops.logsumexp(logits, axes=(1,))
        one_hot_y = init.one_hot(num_classes, y, device=y.device)
        return (logsumexp - (logits * one_hot_y).sum(axes=1)).sum() / bs
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(
            init.ones(dim, device=device, dtype=dtype, requires_grad=True)
        )
        self.bias = Parameter(
            init.zeros(dim, device=device, dtype=dtype, requires_grad=True)
        )
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        assert x.shape[1] == self.dim
        batch_size, num_features = x.shape
        if self.training:
            mean = ops.summation(x, axes=0) / batch_size
            self.running_mean = (
                self.running_mean * (1 - self.momentum) + mean.data * self.momentum
            )
            mean = ops.broadcast_to(ops.reshape(mean, (1, num_features)), x.shape)

            var = ops.summation((x - mean) ** 2, axes=0) / batch_size
            self.running_var = (
                self.running_var * (1 - self.momentum) + var.data * self.momentum
            )
            var = ops.broadcast_to(ops.reshape(var, (1, num_features)), x.shape)

            return (x - mean) / ((var + self.eps) ** 0.5) * self.weight + self.bias

        else:
            mean = ops.broadcast_to(
                ops.reshape(self.running_mean, (1, self.dim)), x.shape
            )
            var = ops.broadcast_to(
                ops.reshape(self.running_var, (1, self.dim)), x.shape
            )

            return (x - mean) / (
                (var + self.eps) ** 0.5
            ) * self.weight.data + self.bias.data
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim: int, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        assert len(x.shape) == 2, "LayerNorm1d only supports 2D input"
        assert x.shape[1] == self.dim, "LayerNorm1d input dimension mismatch"
        bs = x.shape[0]
        mean = ops.summation(x, axes=(1,)) / self.dim
        mean = ops.broadcast_to(ops.reshape(mean, (bs, 1)), x.shape)
        var = ops.summation((x - mean) ** 2, axes=(1,)) / self.dim
        var = ops.broadcast_to(ops.reshape(var, (bs, 1)), x.shape)
        z = (x - mean) / ((var + self.eps) ** 0.5)
        return z * self.weight + self.bias
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            mask = init.randb(*x.shape, p=self.p, device=x.device, dtype=x.dtype)
            mask /= 1 - self.p
            return x * mask
        else:
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return x + self.fn(x)
        ### END YOUR SOLUTION
