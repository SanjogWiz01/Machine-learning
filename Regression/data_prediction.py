import numpy as np
from sklearn.linear_model import LinearRegression

# Example data:
# House size (in sq ft) -> Price (in thousands)
X = np.array([[500], [800], [1000], [1200], [1500], [1800]])
y = np.array([50, 80, 100, 120, 150, 180])

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X, y)

# Make a prediction
size = 1300
predicted_price = model.predict([[size]])

print("Linear Regression Model Trained!")
print("For house size", size, "sq ft, predicted price =", predicted_price[0])
