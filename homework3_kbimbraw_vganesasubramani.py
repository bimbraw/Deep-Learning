import numpy as np
from time import time

def one_hot_encoder(y_data):
    num_label = np.max(y_data) + 1
    return np.eye(num_label)[y_data]

# Splitting training data into train and validation
def split_data(X_data, y_data, split_ratio=0.8):
    # Encoding:
    y_data = one_hot_encoder(y_data)
    n = int(split_ratio * len(y_data))
    # print(n)
    X_train, X_val = X_data[:n], X_data[n:]
    y_train, y_val = y_data[:n], y_data[n:]
    return X_train, np.reshape(y_train, (-1, 10)), X_val, np.reshape(y_val, (-1, 10))

def compute_CE(y, y_hat):
    cross_entropy = 0.0
    # print(y_hat)
    for row in range(y_hat.shape[0]):
        cross_entropy += np.matmul(y[row], np.log(y_hat[row]))
    return cross_entropy

def softmax(z):
    e_pow_Z = np.exp(z)
    # print('e_pow_Z',e_pow_Z.max(), e_pow_Z.min())
    e_ZSum = np.sum(e_pow_Z, axis=1)
    e_ZSum = e_ZSum.reshape((len(e_ZSum), 1))
    # print('e_ZSum',e_ZSum.max(), e_ZSum.min())
    softmax = e_pow_Z / e_ZSum
    return softmax

def gradient(X, y, y_hat):
    grad_w = 1 / len(X) * np.matmul(X.T, (y_hat - y))
    grad_b = 1 / len(X) * np.sum(y_hat - y)
    return grad_w, grad_b

def Softmax_SGD(X_data, y_data, numEpochs, batch_size, lr, alpha):
    w = np.ones((X_data.shape[1], y_data.shape[1]))  # 784 x 10
    b = np.ones((1, y_data.shape[1]))  # 1 x 10
    # print('XSHAPE:' ,X_data.shape)
    # print("YSHAPE", y_data.shape)
    for epoch in range(numEpochs):
        print("Epoch", epoch)
        batch_num = 0
        num_iter = X_data.shape[0] / batch_size
        for _ in range(int(num_iter - 1)):
            start_ind = batch_num * batch_size
            end_ind = (batch_num + 1) * batch_size
            # Creating batch
            X = X_data[start_ind: end_ind]
            y = y_data[start_ind: end_ind]
            batch_num += 1
            # Calculating Z:
            z = np.dot(X, w) + b
            # print('Z',z.max(), z.min())
            # print(z.shape)
            y_hat = np.zeros(z.shape)
            y_hat = softmax(z)
            # print(y_hat)

# Calculating Cross Entropy Loss
crossEntropy_batch = compute_CE(y, y_hat)
reg_loss = 0.0
print("Cross Entropy Batch", crossEntropy_batch)
for k in range(y_hat.shape[1]):
    reg_loss += np.matmul(w.T[k], w[:, k])
print("Reg loss", reg_loss)
fce_batch = -1 * crossEntropy_batch / X.shape[0] + (alpha * reg_loss / 2)
print("batch: {0} \tfce_batch: {1}".format(batch_num - 1, fce_batch))
# Gradient
grad_w, grad_b = gradient(X, y, y_hat)
w = w - (lr * grad_w)
b = b - (lr * grad_b)

return w, b

X_tr = np.load("fashion_mnist_train_images.npy") / 255.0
ytr = np.load("fashion_mnist_train_labels.npy")
X_te = np.load("fashion_mnist_test_images.npy") / 255.0
yte = np.load("fashion_mnist_test_labels.npy")
numEpochs = 10
batch_size = 128
lr = 0.001
alpha = 0

best_model = {'epoch': 0,
              'minibatch': 0,
              'learningRate': 0.,
              'regularization': 0.}

X_train, y_train, X_val, y_val = split_data(X_tr, ytr, split_ratio=0.8)
w, b = Softmax_SGD(X_train, y_train, numEpochs, batch_size, lr, alpha)

# %%
numEpochs = [200, 300, 400, 500]
batch_size = [16, 32, 128, 256]
learning_rate = [0.005, 0.001, 0.05, 0.01]
alpha = [0.0001, 0.001, 0.005, 0.01]

def hyperparams_search(X_val, y_val):
    optimized_weights = optimized_weights = np.random.rand(X_tr.shape[1], 10)
    optimized_bias = np.zeros((1, 10))
    fce_min = 10000
    for epoch in numEpochs:
        for batch in batch_size:
            for lr in learning_rate:
                for a in alpha:
                    weights, bias = Softmax_SGD(X_val, y_val, epoch, batch, lr, a)
                    z_val = np.matmul(X_val, weights) + bias
                    yhat_val = np.zeros(z_val.shape)
                    yhat_val = softmax(z_val)
                    crossEntropy_val = compute_CE(y_val, yhat_val)
                    reg_error_val = 0.0
                    for k in range(z_val.shape[1]):
                        reg_error_val += np.matmul(weights.T[k], weights[:, k])


fce_val = -crossEntropy_val / (X_val.shape[0]) + (a * reg_error_val / 2)
if fce_val < fce_min:
    fce_min = fce_val
    best_model['epoch'] = epoch
    best_model['minibatch'] = batch
    best_model['learningRate'] = lr
    best_model['regularization'] = a
    optimized_weights = np.copy(weights)
    optimized_bias = np.copy(bias)

return fce_min, optimized_weights, optimized_bias

fce_minimum, optimized_weights, optimized_bias = hyperparams_search(X_val, y_val)
print(best_model)
print(fce_minimum)

# %%
numEpochs = 10
batch_size = 128
batch_size = 0.001
alpha = 0.01

def test_Model(X_te, yte, optimized_weights, optimized_bias):
    z_test = np.matmul(X_te, optimized_weights) + optimized_bias
    yhat_test = np.zeros(z_test.shape)
    y_pred_test = np.zeros(z_test.shape)
    crossEntropy_test = 0.0
    correct_pred = 0
    yhat_test = softmax(z_test)
    for r in range(len(z_test)):
        a = max(yhat_test[r])
        y_pred_test[r] = np.where(yhat_test[r] == a, 1, 0)

for row in range(yhat_test.shape[0]):
    for col in range(yhat_test.shape[1]):
        crossEntropy_test += yte[row, col] * np.log(yhat_test[row, col])
    if (yte[row] == y_pred_test[row]).all():
        correct_pred += 1
fce_test = -crossEntropy_test / (len(yte))
accuracy = correct_predict / len(yte)
return fce_test, accuracy

fce_test, accuracy = test_Model(X_te, yte, optimized_weights, optimized_bias)
print('Test Error:', fce_test)
print('Accuracy:', accuracy * 100, '%')

# Accuracy and loss for hyperparameter settings - Accuracy: 0.80. Loss: 0.58
# With increasing epochs, the accuracy increases and the loss decreases