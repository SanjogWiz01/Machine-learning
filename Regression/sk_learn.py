# scikit-learn is a machine learning library for Python
# It provides tools for training models, making predictions, and evaluating them

from sklearn.linear_model import LinearRegression
import numpy as np

# Simple example
X = np.array([[1], [2], [3], [4]])
y = np.array([10, 20, 30, 40])

model = LinearRegression()
model.fit(X, y)

print("Prediction for 5:", model.predict([[5]])[0])
print("This shows how scikit-learn trains and predicts.")
