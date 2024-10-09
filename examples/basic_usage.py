from engine.engine import Engine

def basic_usage():
    a = Engine(2.0)
    b = Engine(3.0)
    c = a * b
    d = c + a
    
    # Backpropagate
    d.backward()

    # Display results
    print(f"Gradient of a: {a.grad}")
    print(f"Gradient of b: {b.grad}")
    print(f"Value of d: {d.data}")

if __name__ == "__main__":
    basic_usage()