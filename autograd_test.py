from __future__ import annotations
from typing import Callable
from numpy.typing import NDArray
import numpy as np
from autograd import grad as agrad
import autograd.numpy as anp
from numpy_wrapper import Node, arr_sum, sin


def grad(fun: Callable) -> Callable:
    """
    Creates gradient function for provided :param fun:

    Gradient function works in the following way:
        1. Performs forward pass to build computational graph and compute the output
        2. Traverses graph backwards replacing operations with its derivatives and passing previous gradients to it
    """
    def gradient(X: Node):
        end_node = forward_pass(fun, X)
        assert isinstance(
            end_node.value, float), "Can not differentiate multivariable function"
        return backward_pass(end_node, X.value)
    return gradient


def forward_pass(fun: Callable, X: Node) -> Node:
    """
    Builds a computational graph for passed function.    
    """
    start_node = X
    end_node = fun(start_node)
    return end_node


def backward_pass(node: Node, X: NDArray):
    """
    Performs a backward pass of value from the given Node.
    """
    nodes: list[Node] = [node]
    while node.parent:
        nodes.append(node.parent)
        node = node.parent
    return np.prod([n.derivative() for n in nodes if n.derivative() is not None], axis=0)


def test(X: Node):
    return -sin(arr_sum(X))


def agrad_test(X: NDArray):
    return -anp.sin(anp.sum(X))


if __name__ == '__main__':
    X = Node(np.array([1., 2., 3.]), np.zeros_like)
    gradient = grad(test)
    a_gradient = agrad(agrad_test)
    print(
        f'Cross entropy: {test(X).value}, custom gradient: {gradient(X)}, autograd gradient: {a_gradient(X.value)}')
