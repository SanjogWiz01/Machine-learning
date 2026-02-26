from sklearn.metrics import mean_squared_error

pred = model.predict(X)

print(mean_squared_error(Y,pred))