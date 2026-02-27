from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures # importing the polynomial freatures to the code

X = [[1], [2], [3], [4], [5]]
y = [2, 6, 14, 28, 45] # values 

poly = PolynomialFeatures(degree=2) # making the polynomila of degree 2
X_poly = poly.fit_transform(X) # i alos donot understadn this 

model = LinearRegression() # evaluating the model is linear regressions 
model.fit(X_poly, y) # fitting the model by the value of x an y.

print("Prediction:", model.predict(poly.transform([[6]]))[0]) # 67