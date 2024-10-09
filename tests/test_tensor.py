from engine.engine import Engine
import numpy as np

def test_engine_operations():
    a = Engine(2.0)
    b = Engine(3.0)
    c = a * b
    d = c + a
    e = d - b
    f = e / a
    g = f ** 2
    
    g.backward()
    
    assert a.grad is not None, "Gradient of a should not be None"
    assert b.grad is not None, "Gradient of b should not be None"
    assert isinstance(a.grad, np.ndarray), "Gradient of a should be a numpy array"
    assert isinstance(b.grad, np.ndarray), "Gradient of b should be a numpy array"

if __name__ == "__main__":
    test_engine_operations()
    print("All tests passed!")