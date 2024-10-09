from engine.engine import Engine

def test_engine_operations():
    a = Engine(2.0)
    b = Engine(3.0)
    c = a * b
    d = c + a
    
    d.backward()
    
    assert a.grad == 4.0, f"Expected gradient of a to be 4.0, but got {a.grad}"
    assert b.grad == 2.0, f"Expected gradient of b to be 2.0, but got {b.grad}"

if __name__ == "__main__":
    test_engine_operations()
    print("All tests passed!")
    