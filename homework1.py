import numpy as np

A = np.array([1, 4, 6])
B = np.array([4, 0.4, -4, 1])
C = np.array([-5.2, 21.6])

A = A.reshape(len(A), 1)
B = B.reshape(len(B), 1)
C = C.reshape(len(C), 1)

print(A)
print(A.shape)

len_val = np.absolute(len(A)-len(B))
append_arr = np.zeros(len_val)
if len(A) > len(B):
    B = np.append(B, append_arr)
    B = B.reshape(len(B), 1)
else:
    A = np.append(A, append_arr)
    A = A.reshape(len(A), 1)
print('This is new A -')
print(A)
print('This is new B -')
print(B)

def problem_1a (A, B):
    add = np.add(A, B)
    return add

print('The solution for problem 1a is -')
print(problem_1a(A, B))

def problem_1b (A, B, C):
    return (np.dot(A, B) - C)

print('The solution for problem 2a is -')
problem_1b(A, B, C)

def problem_1c (A, B, C):
    return ...

def problem_1d (x, y):
    return ...

def problem_1e (A):
    return ...

def problem_1f (A, x):
    return ...

def problem_1g (A, x):
    return ...

def problem_1h (A, alpha):
    return ...

def problem_1i (A, i):
    return ...

def problem_1j (A, c, d):
    return ...

def problem_1k (A, k):
    return ...

def problem_1l (x, k, m, s):
    return ...

def problem_1m (A):
    return ...

def problem_1n (x):
    return ...

def problem_1o (x, k):
    return ...

def problem_1p (X):
    return ...

def linear_regression (X_tr, y_tr):
    ...

def train_age_regressor ():
    # Load data
    X_tr = np.reshape(np.load("age_regression_Xtr.npy"), (-1, 48*48))
    ytr = np.load("age_regression_ytr.npy")
    X_te = np.reshape(np.load("age_regression_Xte.npy"), (-1, 48*48))
    yte = np.load("age_regression_yte.npy")

    w = linear_regression(X_tr, ytr)

    # Report fMSE cost on the training and testing data (separately)
    # ...
