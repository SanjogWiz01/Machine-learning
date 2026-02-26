from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
x=np.random.rand(20,1) * 10
y= 4*x-5
model = LinearRegression()
model.fit(x,y)
model.predict([[4]])