from sklearn.linear_model import LinearRegression

X = [[2, 70], [3, 80], [4, 85], [5, 90]]
y = [60, 65, 70, 78]

model = LinearRegression()
model.fit(X, y)

print("Prediction:", model.predict([[6, 95]])[0])
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)