import numpy as np
from sklearn.linear_model import LinearRegression

# Data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Build model
model = LinearRegression()

# Train model
model.fit(X, y)

# Predict
pred = model.predict([[6]])

print("Model trained!")
print("Prediction for 6:", pred[0])
