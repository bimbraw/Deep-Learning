import numpy as np

A = np.array([1, 4, 6])
B = np.array([4, 0.4, -4, 1])
C = np.array([-5.2, 21.6])

A = A.reshape(len(A), 1)
B = B.reshape(len(B), 1)
C = C.reshape(len(C), 1)

def problem_1a (A, B):
    len_val = np.absolute(len(A) - len(B))
    append_arr = np.zeros(len_val)
    if len(A) > len(B):
        B = np.append(B, append_arr)
        B = B.reshape(len(B), 1)
    else:
        A = np.append(A, append_arr)
        A = A.reshape(len(A), 1)
    add = np.add(A, B)
    return add

print('The solution for problem 1a is -')
print(problem_1a(A, B))

def problem_1b (A, B, C):
    dot_product = np.dot(A, B.T)
    padded_shape = dot_product.shape
    padded_base = np.zeros((padded_shape))
    padded_base[:C.shape[0],:C.shape[1]] = C
    return (dot_product - padded_base)

print('The solution for problem 1b is -')
print(problem_1b(A, B, C))

def problem_1c (A, B, C):
    len_val = np.absolute(len(A) - len(B))
    append_arr = np.zeros(len_val)
    if len(A) > len(B):
        B = np.append(B, append_arr)
        B = B.reshape(len(B), 1)
    else:
        A = np.append(A, append_arr)
        A = A.reshape(len(A), 1)
    hadamard = A * B
    print('This is hadamard shape -')
    print(hadamard.shape)
    print('This is the C.T shape -')
    print((C.T).shape)
    rows = max(hadamard.shape[0], (C.T).shape[0])
    columns = max(hadamard.shape[1], (C.T).shape[1])
    padded_shape = (rows, columns)
    print('padded shape -')
    print(padded_shape)
    padded_base = np.zeros(padded_shape)
    print('padded base -')
    print(padded_base)
    padded_base_hadamard = padded_base
    padded_base_c_trans = padded_base
    padded_base_hadamard[:hadamard.shape[0], :hadamard.shape[1]] = hadamard
    padded_base_c_trans[:(C.T).shape[0], :(C.T).shape[1]] = C.T
    return (padded_base_hadamard + padded_base_c_trans)

print('The solution for problem 1c is -')
print(problem_1c(A, B, C))

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
