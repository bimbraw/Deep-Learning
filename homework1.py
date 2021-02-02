import numpy as np

A = np.array([[1, 4, 6], [-9, 4, 5], [0.1, -0.1, 2]])
B = np.array([[4, -1.1, 7], [2, -0.1, 8], [2, -1, 3]])
C = np.array([[-5.2, 21.6, 3], [5, 6, 7], [-2.3, 1.5, 7]])

#A = A.reshape(len(A), 1)
#B = B.reshape(len(B), 1)
#C = C.reshape(len(C), 1)

def problem_1a (A, B):
    return A + B

print('The solution for problem 1a is -')
print(problem_1a(A, B))

def problem_1b (A, B, C):
    return np.dot(A, B) - C

print('The solution for problem 1b is -')
print(problem_1b(A, B, C))

def problem_1c (A, B, C):
    hadamard = A * B
    return hadamard + C.T

print('The solution for problem 1c is -')
print(problem_1c(A, B, C))

x = np.array([[1, 4], [-4, 1.1]])
y = np.array([[1.12, .4], [-1.4, .1]])

def problem_1d (x, y):
    return np.inner(x, y)

print('The solution for problem 1d is -')
print(problem_1d(x, y))

A = np.random.rand(2, 2)

def problem_1e (A):
    rows = A.shape[0]
    columns = A.shape[1]
    return np.zeros((rows, columns))

print('The solution for problem 1e is -')
print(problem_1e(A))

def problem_1f (A, x):
    return np.linalg.solve(A, x)

print('The solution for problem 1f is -')
print(problem_1f(A, x))

def problem_1g (A, x):
    return np.transpose(np.linalg.solve(x, A))

print('The solution for problem 1g is -')
print(problem_1g(A, x))

alpha = 0.1

def problem_1h (A, alpha):
    return A + (alpha * np.eye(A.shape[0], A.shape[1]))

print('The solution for problem 1h is -')
print(problem_1h(A, alpha))

i = A.shape[0] - 1

#maybe not right
def problem_1i (A, i):
    return np.sum(A[i,::2])

print(A)
print('The solution for problem 1i is -')
print(problem_1i(A, i))

#not completed
def problem_1j (A, c, d):
    return ...

#not completed
def problem_1k (A, k):
    eval, evec = np.linalg.eig(A)
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
