import numpy as np

def linear_regression (X_tr, y_tr):
    X_tr = X_tr.T
    XXt = np.matmul(X_tr, X_tr.T)
    Xy = np.matmul(X_tr, y_tr)
    w = np.linalg.solve(XXt, Xy)
    return w

def fMSE(x, w, y):
    #print(w)
    x = x.T
    y_hat = np.matmul(x.T, w)
    cost = (np.sum(np.square(y_hat - y)))/(2*len(y))
    return cost

def train_age_regressor ():
    # Load data
    X_tr = np.reshape(np.load("age_regression_Xtr.npy"), (-1, 48*48))
    print(X_tr)
    print(X_tr.shape)
    ytr = np.load("age_regression_ytr.npy")
    print(ytr)
    print(ytr.shape)
    X_te = np.reshape(np.load("age_regression_Xte.npy"), (-1, 48*48))
    print(X_te)
    print(X_te.shape)
    yte = np.load("age_regression_yte.npy")
    print(yte)
    print(yte.shape)

    w = linear_regression(X_tr, ytr)
    print(w)

    print("Train MSE is: {:.2f}".format(fMSE(X_tr, w, ytr)))
    print("Test MSE is: {:.2f}".format(fMSE(X_te, w, yte)))

train_age_regressor()