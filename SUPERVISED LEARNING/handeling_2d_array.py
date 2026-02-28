from sklearn.preprocessing import OneHotEncoder
import numpy as np

categorical_feature = ['cat', 'dog', 'dog', 'cat', 'bird']

categorical_feature = np.array(categorical_feature).reshape(-1, 1)

encoder = OneHotEncoder(sparse_output=False)

encoded_feature = encoder.fit_transform(categorical_feature)

print("OneHotEncoded feature:\n", encoded_feature)