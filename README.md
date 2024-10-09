# Autograd Engine
A simple automatic differentiation engine inspired by PyTorch's `autograd`. This engine helps in calculating gradients automatically by constructing and backpropagating through a computation graph.

## Project Structure
```
autograd_engine/
├── engine/
│   ├── __init__.py
│   └── engine.py
├── examples/
│   ├── __init__.py
│   └── basic_usage.py
├── tests/
│   ├── __init__.py
│   └── test_engine.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Getting Started
### Installation
1. Clone the repository:
   ```sh
   git clone <repository_url>
   cd autograd_engine
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   pip install -e .
   ```

### Usage
Run the basic example:
```sh
python examples/basic_usage.py
```

### Testing
Run tests to verify the engine:
```sh
python tests/test_engine.py
```

## Features
- Automatic differentiation
- Backpropagation for gradients
- Supports basic arithmetic operations like addition and multiplication

## Future Improvements
- Add more operations like division, power, and activation functions.
- Implement support for matrix operations and neural network modules.

The README content has been removed. Now, let me explain how the autograd engine works.

## How the Engine Works

The core component of this project is the `Engine` class, which is a simplified implementation of an automatic differentiation engine, like PyTorch’s `autograd`. This class allows you to perform basic operations on tensors (e.g., addition, multiplication) and automatically compute gradients.

#### Key Concepts:

1. **Tensor Representation (`Engine` Class)**:
   - The `Engine` class represents a tensor (an array of numerical data) and tracks operations performed on it.
   - It contains attributes for storing:
     - `data`: The numerical data.
     - `requires_grad`: A boolean indicating if the tensor should track gradients.
     - `grad`: Stores the computed gradient for the tensor (initialized as `None` if gradients are not required).
     - `_backward`: A function that computes the gradient for the tensor.
     - `_prev`: A list of previous nodes (inputs) in the computation graph.

2. **Computation Graph**:
   - When performing operations on instances of `Engine` (e.g., addition, multiplication), the engine constructs a computation graph, where each node is an operation involving one or more tensors.
   - This computation graph is used during the backward pass to propagate gradients.

3. **Operations (`__add__` and `__mul__`)**:
   - The class implements the `+` and `*` operators, which enable you to add or multiply tensors.
   - Each operation creates a new tensor, tracks its inputs, and defines how to compute gradients for each input during backpropagation.
   - For example, for multiplication (`z = x * y`):
     - The gradient of `z` with respect to `x` is `y`.
     - The gradient of `z` with respect to `y` is `x`.

4. **Backpropagation (`backward()` Method)**:
   - The `backward()` method computes gradients for all tensors involved in producing a given output.
   - It uses the chain rule to propagate gradients through the computation graph.
   - The method:
     - Starts by setting the gradient of the output tensor to 1 (since we're interested in the rate of change of the output).
     - Traverses the computation graph in reverse (using a topological sorting method) to compute gradients for each input tensor.

#### Example:
In the usage example (`basic_usage()` function):
- You create two tensors `a` and `b` with values `2.0` and `3.0`.
- You perform operations:
  - `c = a * b` (multiplication)
  - `d = c + a` (addition)
- After calling `d.backward()`, the gradients are computed:
  - `a.grad` will be `4.0` because it is involved in both the multiplication (`grad = b.data`) and addition (`grad = 1`).
  - `b.grad` will be `2.0` because it is only involved in the multiplication with `a`.

The `test_engine_operations()` function in the test file verifies these computed gradients.

This setup provides a foundational understanding of how backpropagation works and serves as a basic building block for creating more complex models, such as neural networks.

## License
This project is licensed under the MIT License.
