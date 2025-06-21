import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("fruits.csv")
X = data[['length_cm', 'weight_g', 'yellow_score']].values
y = data['label'].values.reshape(-1, 1)

# Normalize features
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Loss function
def compute_loss(y_true, y_pred):
    eps = 1e-8  # small value to avoid log(0)
    return -np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))

# Accuracy function
def accuracy(y_true, y_pred):
    return np.mean((y_pred > 0.5) == y_true)

# Initialize parameters
np.random.seed(42)
W = np.random.randn(X.shape[1], 1)
b = 0
lr = 0.1
losses = []
accuracies = []

# Training loop
print("🚀 Starting Training...\n")
for epoch in range(1, 501):
    z = X @ W + b
    y_pred = sigmoid(z)
    loss = compute_loss(y, y_pred)
    acc = accuracy(y, y_pred)

    # Print progress every 50 epochs
    if epoch % 50 == 0 or loss < 0.05:
        print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Accuracy: {acc:.4f}")

    # Gradient calculation
    dz = y_pred - y
    dW = (X.T @ dz) / len(X)
    db = np.mean(dz)

    # Update weights
    W -= lr * dW
    b -= lr * db

    # Track loss and accuracy
    losses.append(loss)
    accuracies.append(acc)

    # Early stopping
    if loss < 0.05:
        print(f"\n✅ Early stopping at epoch {epoch} (Loss < 0.05)\n")
        break

print("🎯 Training Complete!")
print(f"\nFinal Accuracy: {acc:.2f}")
print(f"Final Loss    : {loss:.4f}")

# Plotting results
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(losses, label='Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss over Epochs")
plt.grid(True)
plt.legend()

plt.subplot(1,2,2)
plt.plot(accuracies, label='Accuracy', color='green')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy over Epochs")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
