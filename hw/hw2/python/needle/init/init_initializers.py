import math
from .init_basic import *


def xavier_uniform(fan_in: int, fan_out: int, gain: float = 1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    a = gain * math.sqrt(6.0 / (fan_in + fan_out))
    shape = (fan_in, fan_out)
    return rand(*shape, low=-a, high=a, **kwargs)
    ### END YOUR SOLUTION


def xavier_normal(fan_in: int, fan_out: int, gain: float = 1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    shape = (fan_in, fan_out)
    return randn(*shape, std=std, **kwargs)
    ### END YOUR SOLUTION


def kaiming_uniform(fan_in: int, fan_out: int, nonlinearity: str = "relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    gain = math.sqrt(2.0)
    bound = gain * math.sqrt(3.0 / fan_in)
    shape = (fan_in, fan_out)
    return rand(*shape, low=-bound, high=bound, **kwargs)
    ### END YOUR SOLUTION


def kaiming_normal(fan_in: int, fan_out: int, nonlinearity: str = "relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    gain = math.sqrt(2.0)
    std = gain / math.sqrt(fan_in)
    shape = (fan_in, fan_out)
    return randn(*shape, std=std, **kwargs)
    ### END YOUR SOLUTION
