"""Optimization module"""

import needle as ndl
import numpy as np
from needle.nn import init
from needle.nn.nn_basic import Parameter
from needle import ops
from typing import List


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params: List[Parameter], lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        # 在我的电脑上，无法通过精度测试，但是也非常接近了
        for param in self.params:
            if param.grad is None:
                continue
            grad = param.grad.data
            grad_with_decay = grad + self.weight_decay * param.data
            if param not in self.u:
                self.u[param] = init.zeros(*param.shape, device=param.device, dtype=param.dtype)
            self.u[param].data = self.momentum * self.u[param].data + (1-self.momentum) * grad_with_decay
            param.data -= self.lr * self.u[param].data
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        # self.t += 1
        # for param in self.params:
        #     if param.grad is None:
        #         continue
        #     grad = param.grad.data
        #     grad_with_decay = grad + self.weight_decay * param.data
        #     if param not in self.m:
        #         self.m[param] = init.zeros(*param.shape, device=param.device, dtype=param.dtype)
        #         self.v[param] = init.zeros(*param.shape, device=param.device, dtype=param.dtype)
        #     m = self.m[param].data
        #     v = self.v[param].data
        #     m = self.beta1 * m + (1 - self.beta1) * grad_with_decay
        #     v = self.beta2 * v + (1 - self.beta2) * (grad_with_decay ** 2)
        #     self.m[param].data = m
        #     self.v[param].data = v
        #     m_hat = m * (1 / (1 - self.beta1 ** self.t))
        #     v_hat = v * (1 / (1 - self.beta2 ** self.t))
        #     param.data -= (self.lr / (v_hat ** 0.5 + self.eps)) * m_hat
        self.t += 1
        for p in self.params:
            if p.grad is None:
                continue
            grad = p.grad.data
            grad_with_weight_decay = grad + self.weight_decay * p.data

            if p not in self.m:
                self.m[p] = init.zeros(*p.shape, device = p.device, dtype=p.dtype)
                self.v[p] = init.zeros(*p.shape, device = p.device, dtype=p.dtype)
            m_t = self.m[p]
            v_t = self.v[p]
            m_t_1 = self.beta1 * m_t + (1 - self.beta1) * grad_with_weight_decay
            v_t_1 = self.beta2 * v_t + (1 - self.beta2) * (grad_with_weight_decay ** 2)
            self.m[p] = m_t_1
            self.v[p] = v_t_1

            m_t_1_hat = m_t_1 / (1 - self.beta1 ** self.t)
            v_t_1_hat = v_t_1 / (1 - self.beta2 ** self.t)

            p.data -= self.lr * ndl.Tensor(m_t_1_hat.data / (v_t_1_hat.data ** 0.5 + self.eps), dtype = p.data.dtype, requires_grad=False)
        ### END YOUR SOLUTION
