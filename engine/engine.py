import numpy as np

class Engine:
    def __init__(self, data, requires_grad=True):
        self.data = np.array(data, dtype=float)  # Wrap the data in a numpy array to handle different data types
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None  # Gradient initialized to zero if required
        self._backward = lambda: None  # A function for calculating the gradient
        self._prev = []  # Previous nodes in the computation graph

    def backward(self):
        # Set gradient of output node (root of the graph) to 1
        self.grad = np.ones_like(self.data)
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for parent in v._prev:
                    build_topo(parent)
                topo.append(v)

        build_topo(self)

        for v in reversed(topo):
            v._backward()

    def __add__(self, other):
        other = other if isinstance(other, Engine) else Engine(other)
        out = Engine(self.data + other.data)

        def _backward():
            if self.requires_grad:
                self.grad += out.grad
            if other.requires_grad:
                other.grad += out.grad

        out._backward = _backward
        out._prev = [self, other]
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Engine) else Engine(other)
        out = Engine(self.data - other.data)

        def _backward():
            if self.requires_grad:
                self.grad += out.grad
            if other.requires_grad:
                other.grad -= out.grad

        out._backward = _backward
        out._prev = [self, other]
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Engine) else Engine(other)
        out = Engine(self.data * other.data)

        def _backward():
            if self.requires_grad:
                self.grad += other.data * out.grad
            if other.requires_grad:
                other.grad += self.data * out.grad

        out._backward = _backward
        out._prev = [self, other]
        return out

    def __truediv__(self, other):
        other = other if isinstance(other, Engine) else Engine(other)
        out = Engine(self.data / other.data)

        def _backward():
            if self.requires_grad:
                self.grad += out.grad / other.data
            if other.requires_grad:
                other.grad -= (self.data / (other.data ** 2)) * out.grad

        out._backward = _backward
        out._prev = [self, other]
        return out

    def __pow__(self, exponent):
        out = Engine(self.data ** exponent)

        def _backward():
            if self.requires_grad:
                self.grad += (exponent * (self.data ** (exponent - 1))) * out.grad

        out._backward = _backward
        out._prev = [self]
        return out

    def matmul(self, other):
        other = other if isinstance(other, Engine) else Engine(other)
        out = Engine(np.dot(self.data, other.data))

        def _backward():
            if self.requires_grad:
                self.grad += np.dot(out.grad, other.data.T)
            if other.requires_grad:
                other.grad += np.dot(self.data.T, out.grad)

        out._backward = _backward
        out._prev = [self, other]
        return out

    def __repr__(self):
        return f"Engine(data={self.data}, requires_grad={self.requires_grad}, grad={self.grad})"
