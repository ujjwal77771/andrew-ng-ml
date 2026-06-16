import numpy as np

from normalization import zscore_normalize_features
from gradient_descent import gradient_descent

# sample data
X = np.array([
    [2104, 5, 1, 45],
    [1416, 3, 2, 40],
    [852, 2, 1, 35],
    [1940, 4, 2, 20]
], dtype=float)

y = np.array([460, 232, 178, 315], dtype=float)

# normalize
X_norm, mu, sigma = zscore_normalize_features(X)

# initialize
w = np.zeros(X.shape[1])
b = 0.0

# train
w, b, history = gradient_descent(
    X_norm,
    y,
    w,
    b,
    alpha=0.1,
    iterations=1000
)

print("w =", w)
print("b =", b)

# predict first example
x = (X[0] - mu) / sigma

prediction = np.dot(x, w) + b

print("Prediction:", prediction)
print("Actual:", y[0])