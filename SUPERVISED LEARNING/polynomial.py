from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

X = [[1], [2], [3], [4], [5]]
y = [2, 6, 14, 28, 45]

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

print("Prediction:", model.predict(poly.transform([[6]]))[0])