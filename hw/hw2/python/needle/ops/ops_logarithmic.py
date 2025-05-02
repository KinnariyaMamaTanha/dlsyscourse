from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api


class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        assert len(Z.shape) == 2, "LogSoftmax only supports 2D tensors"
        Z_max = array_api.max(Z, axis=1, keepdims=True)
        return (
            Z
            - Z_max
            - array_api.log(
                array_api.sum(array_api.exp(Z - Z_max), axis=1, keepdims=True)
            )
        )
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        input = node.inputs[0]
        input_shape = input.shape
        axes = (1,)
        reshape_shape = [input_shape[0], 1]
        input_max = Tensor(input.numpy().max(axis=axes), device=input.device)
        input_max = broadcast_to(reshape(input_max, reshape_shape), input_shape)
        input_minus_max = input - input_max
        input_exp = exp(input_minus_max)
        input_sum_exp = broadcast_to(
            reshape(summation(input_exp, axes), reshape_shape), input_shape
        )
        softmax_val = input_exp / input_sum_exp
        sum_out_grad = summation(out_grad, axes)
        sum_out_grad_reshaped = reshape(sum_out_grad, reshape_shape)
        sum_out_grad_broadcasted = broadcast_to(sum_out_grad_reshaped, input_shape)
        return out_grad - softmax_val * sum_out_grad_broadcasted
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        Z_max = array_api.max(Z, axis=self.axes, keepdims=True)
        reshape_shape = []
        for i in range(len(Z.shape)):
            if Z_max.shape[i] != 1:
                reshape_shape.append(Z.shape[i])
        Z = (
            array_api.log(
                array_api.sum(array_api.exp(Z - Z_max), axis=self.axes, keepdims=True)
            )
            + Z_max
        )
        Z = array_api.reshape(Z, tuple(reshape_shape))
        return Z
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        input = node.inputs[0]
        input_shape = input.shape
        if self.axes is None:
            reshape_shape = [1 for _ in input_shape]
        else:
            reshape_shape = [
                1 if i in self.axes else input_shape[i] for i in range(len(input_shape))
            ]
        input_max = Tensor(input.numpy().max(axis=self.axes), device=input.device)
        input_max = broadcast_to(reshape(input_max, reshape_shape), input_shape)
        input_minus_max = input - input_max
        input_exp = exp(input_minus_max)
        input_sum_exp = broadcast_to(
            reshape(summation(input_exp, self.axes), reshape_shape), input_shape
        )
        return broadcast_to(reshape(out_grad, reshape_shape), input_shape) * (
            input_exp / input_sum_exp
        )
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)
