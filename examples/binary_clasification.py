import numpy as np
import matplotlib.pyplot as plt
from engine.engine import Engine

plt.style.use(plt.style.available[10])

# Set the random seed for reproducibility
np.random.seed(42)

# Generate x-coordinates for both classes
x = np.linspace(-1.5, 1, 200)

# Generate y-coordinates for class 0 (lower wave)
y0 = np.cos(4 * x) + np.random.normal(0, 0.2, 200)

# Generate y-coordinates for class 1 (upper wave)
y1 = np.sin(2 * x) + np.random.normal(3, 0.2, 200)

# Combine the coordinates into a single dataset
X = np.vstack((np.column_stack((x, y0)), np.column_stack((x, y1))))

# Create labels for the dataset
y = np.hstack((np.zeros(200), np.ones(200)))

# Plot the dataset
plt.figure(figsize=(8, 6))
plt.scatter(X[:200, 0], X[:200, 1], color='green', label='Class 0')
plt.scatter(X[200:, 0], X[200:, 1], color='gray', label='Class 1')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Random Binary Classification Dataset (Wave Pattern)')
plt.legend()
plt.grid(True)
plt.show()

# Define the Sequential Model using Engine class
class SimpleNeuralNetwork:
    def __init__(self):
        # Initialize weights and biases
        self.w1 = Engine(np.random.randn(), requires_grad=True)
        self.w2 = Engine(np.random.randn(), requires_grad=True)
        self.b = Engine(np.random.randn(), requires_grad=True)

    def forward(self, x1, x2):
        # Simple linear combination with sigmoid activation
        z = self.w1 * x1 + self.w2 * x2 + self.b
        return 1 / (1 + np.exp(-z.data))  # Sigmoid activation

# Training the simple neural network
model = SimpleNeuralNetwork()
learning_rate = 0.01
epochs = 100

losses = []
for epoch in range(epochs):
    total_loss = 0
    for i in range(len(X)):
        x1 = Engine(X[i, 0], requires_grad=False)
        x2 = Engine(X[i, 1], requires_grad=False)
        target = y[i]

        # Forward pass
        pred = model.forward(x1, x2)

        # Binary cross-entropy loss
        loss = -(target * np.log(pred) + (1 - target) * np.log(1 - pred))
        total_loss += loss

        # Backward pass
        grad = pred - target  # Derivative of loss w.r.t. prediction
        model.w1.grad = grad * x1.data
        model.w2.grad = grad * x2.data
        model.b.grad = grad

        # Update weights
        model.w1.data -= learning_rate * model.w1.grad
        model.w2.data -= learning_rate * model.w2.grad
        model.b.data -= learning_rate * model.b.grad

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

# Decision Boundary Visualization
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

Z = np.array([model.forward(Engine(x, requires_grad=False), Engine(y, requires_grad=False)) for x, y in zip(xx.ravel(), yy.ravel())])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.8, cmap='BuGn')
plt.scatter(X[:200, 0], X[:200, 1], color='green', label='Class 0')
plt.scatter(X[200:, 0], X[200:, 1], color='gray', label='Class 1')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Decision Boundary for Wave Pattern Classification')
plt.legend()
plt.grid(True)
plt.show()