from engine.engine import Engine

def basic_usage():
    a = Engine(4.0)
    b = Engine(5.0)
    c = a * b
    d = c + a
    e = d - b
    f = e / a
    g = f ** 2
    h = a.matmul(b)  # Assuming 'a' and 'b' are compatible matrices
    
    # Backpropagate
    g.backward()

    # Display results
    print(f"Gradient of a: {a.grad}")
    print(f"Gradient of b: {b.grad}")
    print(f"Value of g: {g.data}")

if __name__ == "__main__":
    basic_usage()