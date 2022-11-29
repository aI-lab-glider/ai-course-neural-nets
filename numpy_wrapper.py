"""
Module contains wrappers for numpy functions and makes the autodifferentiation possible
"""
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Callable
from abc import ABC, abstractmethod
import functools


class Node:
    def __init__(self, value: NDArray, derivative_fun: Callable[[NDArray], NDArray], parent: Node | None = None) -> None:
        self.value = value
        self.parent = parent
        self.derivative_function = derivative_fun

    def derivative(self):
        return self.derivative_function(self.parent.value)\
            if self.parent is not None\
            else None

    def __neg__(self):
        return Node(-self.value, lambda X: -np.ones_like(X), self)


def primitive(deriv):
    def primitive_wrapper(fun):
        @functools.wraps(fun)
        def computation(X: Node) -> Node:
            val = X.value
            computation_result = fun(val)
            return Node(computation_result, deriv, X)
        return computation
    return primitive_wrapper


@primitive(deriv=np.ones_like)
def arr_sum(X: NDArray):
    return np.sum(X)


@primitive(deriv=np.cos)
def sin(X: NDArray):
    return np.sin(X)
