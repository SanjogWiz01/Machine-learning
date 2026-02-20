import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. Create dataset (hours studied -> marks)
data = {
    "hours": [1, 2, 3, 4, 5, 6, 7, 8],
    "marks": [40, 45, 50, 55, 60, 65, 70, 75]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)

# 2. Split into features (X) and target (y)
X = df[["hours"]]   # 2D
y = df["marks"]     # 1D

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# 4. Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Predict on test data
y_pred = model.predict(X_test)

print("\nTest Predictions vs Actual:")
for pred, actual in zip(y_pred, y_test):
    print(f"Predicted: {pred:.2f}, Actual: {actual}")

# 6. Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# 7. Predict for new input
hours = float(input("\nEnter hours studied: "))
predicted_marks = model.predict([[hours]])

print(f"Predicted marks for {hours} hours study: {predicted_marks[0]:.2f}")

# 8. Plot the result
plt.figure()
plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.title("Hours Studied vs Marks (Linear Regression)")
plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.show() # printing the patttern
