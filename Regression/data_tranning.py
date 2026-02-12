import numpy as np
from sklearn.linear_model import LinearRegression

# Simple data: X -> input, y -> output
# Example: hours studied -> marks
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([40, 45, 50, 55, 60])

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Predict new value
hours = 6
prediction = model.predict([[hours]])

print("Trained Linear Regression Model")
print("For", hours, "hours studied, predicted marks =", prediction[0])
