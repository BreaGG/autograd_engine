import numpy as np
import matplotlib.pyplot as plt
from engine.engine import Engine

# Generate synthetic dataset
np.random.seed(42)
X = np.linspace(0, 10, 100)
y = 2.5 * X + np.random.normal(0, 1, 100)

# Plot the dataset
plt.figure(figsize=(8, 6))
plt.scatter(X, y, label='Data', color='blue')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Synthetic Linear Dataset')
plt.legend()
plt.grid(True)
plt.show()

# Initialize weight and bias
w = Engine(np.random.randn(), requires_grad=True)
b = Engine(np.random.randn(), requires_grad=True)
learning_rate = 0.01
epochs = 50

# Training loop
losses = []
for epoch in range(epochs):
    total_loss = 0
    for i in range(len(X)):
        x = Engine(X[i], requires_grad=False)
        target = y[i]

        # Forward pass: Linear model y = w * x + b
        y_pred = w * x + b

        # Loss: Mean Squared Error
        loss = (y_pred - target) ** 2
        total_loss += loss.data

        # Backward pass
        loss.backward()

        # Update weights
        w.data -= learning_rate * w.grad
        b.data -= learning_rate * b.grad

        # Zero gradients
        w.grad = np.zeros_like(w.data)
        b.grad = np.zeros_like(b.data)

    avg_loss = total_loss / len(X)
    losses.append(avg_loss)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss}")

# Plot the loss over epochs
plt.figure(figsize=(8, 6))
plt.plot(losses, label='Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.grid(True)
plt.show()

# Plot the fitted line
plt.figure(figsize=(8, 6))
plt.scatter(X, y, label='Data', color='blue')
plt.plot(X, w.data * X + b.data, label='Fitted Line', color='red')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression Fit')
plt.legend()
plt.grid(True)
plt.show()
