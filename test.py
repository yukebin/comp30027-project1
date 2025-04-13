import numpy as np

y = np.array([0, 1, 0, 1])
X = np.array([
 [1, 0, 2],  # message 0 (class 0)
 [0, 1, 0],  # message 1 (class 1)
 [1, 1, 0],  # message 2 (class 0)
 [0, 0, 3]   # message 3 (class 1)
])

# Get just class 0 rows
X_c = X[y == 0]  # rows 0 and 2
print(X_c)