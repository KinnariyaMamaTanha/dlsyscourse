"""Core data structures."""
import needle
from .backend_numpy import Device, cpu, all_devices
from typing import List, Optional, NamedTuple, Tuple, Union
from collections import namedtuple
import numpy

from needle import init

from typing import Dict

# needle version
LAZY_MODE = False
TENSOR_COUNTER = 0

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks
import numpy as array_api

NDArray = numpy.ndarray


class Op:
    """Operator definition."""

    def __call__(self, *args):
        raise NotImplementedError()

    def compute(self, *args: Tuple[NDArray]):
        """Calculate forward pass of operator.

        Parameters
        ----------
        input: np.ndarray
            A list of input arrays to the function

        Returns
        -------
        output: nd.array
            Array output of the operation

        """
        raise NotImplementedError()

    def gradient(
        self, out_grad: "Value", node: "Value"
    ) -> Union["Value", Tuple["Value"]]:
        """Compute partial adjoint for each input value for a given output adjoint.

        Parameters
        ----------
        out_grad: Value
            The adjoint wrt to the output value.

        node: Value
            The value node of forward evaluation.

        Returns
        -------
        input_grads: Value or Tuple[Value]
            A list containing partial gradient adjoints to be propagated to
            each of the input node.
        """
        raise NotImplementedError()

    def gradient_as_tuple(self, out_grad: "Value", node: "Value") -> Tuple["Value"]:
        """Convenience method to always return a tuple from gradient call"""
        output = self.gradient(out_grad, node)
        if isinstance(output, tuple):
            return output
        elif isinstance(output, list):
            return tuple(output)
        else:
            return (output,)


class TensorOp(Op):
    """Op class specialized to output tensors, will be alternate subclasses for other structures"""

    def __call__(self, *args):
        return Tensor.make_from_op(self, args)


class TensorTupleOp(Op):
    """Op class specialized to output TensorTuple"""

    def __call__(self, *args):
        return TensorTuple.make_from_op(self, args)


class Value:
    """A value in the computational graph."""

    # trace of computational graph
    op: Optional[Op]
    inputs: List["Value"]
    # The following fields are cached fields for
    # dynamic computation
    cached_data: NDArray # Here NDArray is a numpy array, i.e. numpy.ndarray,
                         # which will be replaced by our own array type in the future
    requires_grad: bool

    def realize_cached_data(self):
        """Run compute to realize the cached data"""
        # avoid recomputation
        if self.cached_data is not None:
            return self.cached_data
        # note: data implicitly calls realized cached data
        self.cached_data = self.op.compute(
            *[x.realize_cached_data() for x in self.inputs]
        )
        return self.cached_data

    def is_leaf(self):
        return self.op is None

    def __del__(self):
        global TENSOR_COUNTER
        TENSOR_COUNTER -= 1

    def _init(
        self,
        op: Optional[Op],
        inputs: List["Tensor"],
        *,
        num_outputs: int = 1,
        cached_data: List[object] = None,
        requires_grad: Optional[bool] = None,
    ):
        global TENSOR_COUNTER
        TENSOR_COUNTER += 1
        if requires_grad is None:
            requires_grad = any(x.requires_grad for x in inputs)
        self.op = op
        self.inputs = inputs
        self.num_outputs = num_outputs
        self.cached_data = cached_data
        self.requires_grad = requires_grad

    @classmethod
    def make_const(cls, data, *, requires_grad=False):
        value = cls.__new__(cls)
        value._init(
            None,
            [],
            cached_data=data,
            requires_grad=requires_grad,
        )
        return value

    @classmethod
    def make_from_op(cls, op: Op, inputs: List["Value"]):
        value = cls.__new__(cls)
        value._init(op, inputs)

        if not LAZY_MODE:
            if not value.requires_grad:
                return value.detach()
            value.realize_cached_data()
        return value


### Not needed in HW1
class TensorTuple(Value):
    """Represent a tuple of tensors.

    To keep things simple, we do not support nested tuples.
    """

    def __len__(self):
        cdata = self.realize_cached_data()
        return len(cdata)

    def __getitem__(self, index: int):
        return needle.ops.tuple_get_item(self, index)

    def tuple(self):
        return tuple([x for x in self])

    def __repr__(self):
        return "needle.TensorTuple" + str(self.tuple())

    def __str__(self):
        return self.__repr__()

    def __add__(self, other):
        assert isinstance(other, TensorTuple)
        assert len(self) == len(other)
        return needle.ops.make_tuple(*[self[i] + other[i] for i in range(len(self))])

    def detach(self):
        """Create a new tensor that shares the data but detaches from the graph."""
        return Tuple.make_const(self.realize_cached_data())


class Tensor(Value):
    """A multi-dimensional array representing a tensor in the computational graph.

    This class extends the Value class to provide tensor operations with automatic
    differentiation capabilities. It supports various mathematical operations,
    shape manipulations, and gradient computation.

    Attributes:
        grad: The gradient of this tensor with respect to some scalar output.
    """

    grad: "Tensor"

    def __init__(
        self,
        array,
        *,
        device: Optional[Device] = None,
        dtype=None,
        requires_grad=True,
        **kwargs,
    ):
        """Initialize a new Tensor.

        Args:
            array: Data to initialize the tensor with. Can be a numpy array,
                  another Tensor, or any array-like object.
            device: The device to store this tensor on (CPU/GPU). Defaults to None,
                   which uses the device of the input tensor or CPU.
            dtype: Data type of the tensor. Defaults to None, which uses the dtype
                  of the input tensor or the default dtype.
            requires_grad: Whether to track gradients for this tensor. Defaults to True.
            **kwargs: Additional keyword arguments.
        """
        if isinstance(array, Tensor):
            if device is None:
                device = array.device
            if dtype is None:
                dtype = array.dtype
            if device == array.device and dtype == array.dtype:
                cached_data = array.realize_cached_data()
            else:
                # fall back, copy through numpy conversion
                cached_data = Tensor._array_from_numpy(
                    array.numpy(), device=device, dtype=dtype
                )
        else:
            device = device if device else cpu()
            cached_data = Tensor._array_from_numpy(array, device=device, dtype=dtype)

        self._init(
            None,
            [],
            cached_data=cached_data,
            requires_grad=requires_grad,
        )

    @staticmethod
    def _array_from_numpy(numpy_array, device, dtype):
        """Convert a numpy array to the appropriate array type based on the backend.

        Args:
            numpy_array: The numpy array to convert.
            device: The device to place the array on.
            dtype: The data type for the array.

        Returns:
            The converted array in the appropriate backend format.
        """
        if array_api is numpy:
            return numpy.array(numpy_array, dtype=dtype)
        return array_api.array(numpy_array, device=device, dtype=dtype)

    @staticmethod
    def make_from_op(op: Op, inputs: List["Value"]):
        """Create a new tensor from an operation and its inputs.

        Args:
            op: The operation that produces this tensor.
            inputs: The input values to the operation.

        Returns:
            A new Tensor resulting from the operation.
        """
        tensor = Tensor.__new__(Tensor)
        tensor._init(op, inputs)
        if not LAZY_MODE:
            if not tensor.requires_grad:
                return tensor.detach()
            tensor.realize_cached_data()
        return tensor

    @staticmethod
    def make_const(data, requires_grad=False):
        """Create a constant tensor from data.

        Args:
            data: The data to create the constant tensor from.
            requires_grad: Whether the constant requires gradients. Defaults to False.

        Returns:
            A new constant Tensor.
        """
        tensor = Tensor.__new__(Tensor)
        tensor._init(
            None,
            [],
            cached_data=data
            if not isinstance(data, Tensor)
            else data.realize_cached_data(),
            requires_grad=requires_grad,
        )
        return tensor

    @property
    def data(self):
        """Get a detached version of this tensor that shares the same data.

        Returns:
            A detached Tensor with the same data.
        """
        return self.detach()

    @data.setter
    def data(self, value):
        """Set the data of this tensor.

        Args:
            value: A Tensor with the new data.

        Raises:
            AssertionError: If value is not a Tensor or has a different dtype.
        """
        assert isinstance(value, Tensor)
        assert value.dtype == self.dtype, "%s %s" % (
            value.dtype,
            self.dtype,
        )
        self.cached_data = value.realize_cached_data()

    def detach(self):
        """Create a new tensor that shares the data but detaches from the graph.

        Returns:
            A new Tensor with the same data but no gradient tracking.
        """
        return Tensor.make_const(self.realize_cached_data())

    @property
    def shape(self):
        """Get the shape of this tensor.

        Returns:
            The shape of the tensor as a tuple.
        """
        return self.realize_cached_data().shape

    @property
    def dtype(self):
        """Get the data type of this tensor.

        Returns:
            The data type of the tensor.
        """
        return self.realize_cached_data().dtype

    @property
    def device(self):
        """Get the device this tensor is stored on.

        Returns:
            The device of the tensor (CPU/GPU).
        """
        data = self.realize_cached_data()
        # numpy array always sits on cpu
        if array_api is numpy:
            return cpu()
        return data.device

    def backward(self, out_grad=None):
        """Compute gradients of this tensor with respect to graph leaves.

        Args:
            out_grad: The gradient of the output with respect to this tensor.
                     If None, defaults to a tensor of ones with the same shape.
        """
        out_grad = (
            out_grad
            if out_grad
            else init.ones(*self.shape, dtype=self.dtype, device=self.device)
        )
        compute_gradient_of_variables(self, out_grad)

    def __repr__(self):
        """Return a string representation of the tensor.

        Returns:
            A string representation of the tensor.
        """
        return "needle.Tensor(" + str(self.realize_cached_data()) + ")"

    def __str__(self):
        """Return a string representation of the tensor data.

        Returns:
            A string representation of the tensor data.
        """
        return self.realize_cached_data().__str__()

    def numpy(self):
        """Convert this tensor to a numpy array.

        Returns:
            A numpy array with the same data as this tensor.
        """
        data = self.realize_cached_data()
        if array_api is numpy:
            return data
        return data.numpy()

    def __add__(self, other):
        """Add another tensor or scalar to this tensor.

        Args:
            other: The tensor or scalar to add.

        Returns:
            A new tensor with the result of the addition.
        """
        if isinstance(other, Tensor):
            return needle.ops.EWiseAdd()(self, other)
        else:
            return needle.ops.AddScalar(other)(self)

    def __mul__(self, other):
        """Multiply this tensor by another tensor or scalar.

        Args:
            other: The tensor or scalar to multiply by.

        Returns:
            A new tensor with the result of the multiplication.
        """
        if isinstance(other, Tensor):
            return needle.ops.EWiseMul()(self, other)
        else:
            return needle.ops.MulScalar(other)(self)

    def __pow__(self, other):
        """Raise this tensor to the power of another tensor or scalar.

        Args:
            other: The tensor or scalar exponent.

        Returns:
            A new tensor with the result of the power operation.
        """
        if isinstance(other, Tensor):
            return needle.ops.EWisePow()(self, other)
        else:
            return needle.ops.PowerScalar(other)(self)

    def __sub__(self, other):
        """Subtract another tensor or scalar from this tensor.

        Args:
            other: The tensor or scalar to subtract.

        Returns:
            A new tensor with the result of the subtraction.
        """
        if isinstance(other, Tensor):
            return needle.ops.EWiseAdd()(self, needle.ops.Negate()(other))
        else:
            return needle.ops.AddScalar(-other)(self)

    def __truediv__(self, other):
        """Divide this tensor by another tensor or scalar.

        Args:
            other: The tensor or scalar to divide by.

        Returns:
            A new tensor with the result of the division.
        """
        if isinstance(other, Tensor):
            return needle.ops.EWiseDiv()(self, other)
        else:
            return needle.ops.DivScalar(other)(self)

    def __matmul__(self, other):
        """Perform matrix multiplication with another tensor.

        Args:
            other: The tensor to multiply with.

        Returns:
            A new tensor with the result of the matrix multiplication.
        """
        return needle.ops.MatMul()(self, other)

    def matmul(self, other):
        """Perform matrix multiplication with another tensor.

        Args:
            other: The tensor to multiply with.

        Returns:
            A new tensor with the result of the matrix multiplication.
        """
        return needle.ops.MatMul()(self, other)

    def sum(self, axes=None):
        """Sum the tensor along specified axes.

        Args:
            axes: The axes to sum over. None means sum over all axes.

        Returns:
            A new tensor with the result of the summation.
        """
        return needle.ops.Summation(axes)(self)

    def broadcast_to(self, shape):
        """Broadcast this tensor to a new shape.

        Args:
            shape: The target shape to broadcast to.

        Returns:
            A new tensor broadcasted to the target shape.
        """
        return needle.ops.BroadcastTo(shape)(self)

    def reshape(self, shape):
        """Reshape this tensor to a new shape.

        Args:
            shape: The target shape to reshape to.

        Returns:
            A new tensor with the same data but reshaped.
        """
        return needle.ops.Reshape(shape)(self)

    def __neg__(self):
        """Negate this tensor.

        Returns:
            A new tensor with all elements negated.
        """
        return needle.ops.Negate()(self)

    def transpose(self, axes=None):
        """Transpose this tensor along specified axes.

        Args:
            axes: The permutation of the dimensions. None means reverse the dimensions.

        Returns:
            A new tensor with the dimensions permuted.
        """
        return needle.ops.Transpose(axes)(self)

    __radd__ = __add__
    __rmul__ = __mul__
    __rsub__ = __sub__
    __rmatmul__ = __matmul__


def compute_gradient_of_variables(output_tensor, out_grad):
    """Take gradient of output node with respect to each node in node_list.

    Store the computed result in the grad field of each Variable.
    """
    # a map from node to a list of gradient contributions from each output node
    node_to_output_grads_list: Dict[Tensor, List[Tensor]] = {}
    # Special note on initializing gradient of
    # We are really taking a derivative of the scalar reduce_sum(output_node)
    # instead of the vector output_node. But this is the common case for loss function.
    node_to_output_grads_list[output_tensor] = [out_grad]

    # Traverse graph in reverse topological order given the output_node that we are taking gradient wrt.
    reverse_topo_order = list(reversed(find_topo_sort([output_tensor])))

    ### BEGIN YOUR SOLUTION
    for node in reverse_topo_order:
        # Accumulate gradients for the current node
        node_grad = sum_node_list(node_to_output_grads_list[node])
        
        if node.is_leaf():
            # For leaf nodes, we store the gradient but don't propagate further
            node.grad = node_grad
            continue
            
        # Compute gradients for each input
        grads = node.op.gradient_as_tuple(node_grad, node)
        
        # Distribute gradients to inputs
        for i, input_node in enumerate(node.inputs):
            grad = grads[i]
            if input_node not in node_to_output_grads_list:
                node_to_output_grads_list[input_node] = [grad]
            else:
                node_to_output_grads_list[input_node].append(grad)
    ### END YOUR SOLUTION


def find_topo_sort(node_list: List[Value]) -> List[Value]:
    """Given a list of nodes, return a topological sort list of nodes ending in them.

    A simple algorithm is to do a post-order DFS traversal on the given nodes,
    going backwards based on input edges. Since a node is added to the ordering
    after all its predecessors are traversed due to post-order DFS, we get a topological
    sort.
    """
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


def topo_sort_dfs(node, visited, topo_order):
    """Post-order DFS"""
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


##############################
####### Helper Methods #######
##############################


def sum_node_list(node_list: List[Value]):
    """Custom sum function in order to avoid create redundant nodes in Python sum implementation."""
    from operator import add
    from functools import reduce

    return reduce(add, node_list)
