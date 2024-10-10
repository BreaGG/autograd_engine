from engine.engine import Engine

def real_example():
    # Let's create a simple example: y = (3 * x^2) + (2 * x) + 1
    x = Engine(5.0, requires_grad=True)
    three = Engine(3.0, requires_grad=False)
    two = Engine(2.0, requires_grad=False)
    one = Engine(1.0, requires_grad=False)

    y = (three * (x ** 2)) + (two * x) + one
    
    # Perform backpropagation to find dy/dx
    y.backward()

    # Display results
    print(f"Value of y: {y.data}")
    print(f"Gradient of x: {x.grad}")

if __name__ == "__main__":
    real_example()
