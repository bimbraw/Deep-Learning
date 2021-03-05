# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from time import time
from sklearn.decomposition import PCA
import random

# For this assignment, assume that every hidden layer has the same number of neurons.
# Optimized Parameters
NUM_HIDDEN_LAYERS = 5
NUM_INPUT = 784
NUM_HIDDEN = 50
NUM_OUTPUT = 10
LAMBDA_W = 0.01
NUM_EPOCHS = 10


# %%
# Unpack a list of weights and biases into their individual np.arrays.
def unpack(weightsAndBiases):
    # Unpack arguments
    Ws = []
    nsum = 0
    # Weight matrices
    start = 0
    end = NUM_INPUT * NUM_HIDDEN
    W = weightsAndBiases[start:end]
    Ws.append(W)
    nsum += len(Ws[-1])
    for i in range(NUM_HIDDEN_LAYERS - 1):
        start = end
        end = end + NUM_HIDDEN * NUM_HIDDEN
        W = weightsAndBiases[start:end]
        Ws.append(W)
        nsum += len(Ws[-1])
    start = end
    end = end + NUM_HIDDEN * NUM_OUTPUT
    W = weightsAndBiases[start:end]
    Ws.append(W)
    nsum += len(Ws[-1])
    Ws[0] = Ws[0].reshape(NUM_INPUT, NUM_HIDDEN)
    for i in range(1, NUM_HIDDEN_LAYERS):
        # Convert from vectors into matrices
        Ws[i] = Ws[i].reshape(NUM_HIDDEN, NUM_HIDDEN)
    Ws[-1] = Ws[-1].reshape(NUM_HIDDEN, NUM_OUTPUT)
    # Bias terms
    bs = []
    start = end
    end = end + NUM_HIDDEN
    b = weightsAndBiases[start:end]
    bs.append(b)
    nsum += len(bs[-1])
    for i in range(NUM_HIDDEN_LAYERS - 1):
        start = end
        end = end + NUM_HIDDEN
        b = weightsAndBiases[start:end]
        bs.append(b)
        nsum += len(bs[-1])
    start = end
    end = end + NUM_OUTPUT
    b = weightsAndBiases[start:end]
    bs.append(b)
    nsum += len(bs[-1])
    return Ws, bs


def initWeightsAndBiases():
    Ws = []
    bs = []
    np.random.seed(0)
    W = 2 * (np.random.random(size=(NUM_INPUT, NUM_HIDDEN)) / NUM_INPUT ** 0.5) - 1. / NUM_INPUT ** 0.5
    Ws.append(W)
    b = 0.01 * np.ones(NUM_HIDDEN)
    bs.append(b)
    for i in range(NUM_HIDDEN_LAYERS - 1):
        W = 2 * (np.random.random(size=(NUM_HIDDEN, NUM_HIDDEN)) / NUM_HIDDEN ** 0.5) - 1. / NUM_HIDDEN ** 0.5
        Ws.append(W)
        bs.append(b)
    W = 2 * (np.random.random(size=(NUM_HIDDEN, NUM_OUTPUT)) / NUM_HIDDEN ** 0.5) - 1. / NUM_HIDDEN ** 0.5
    Ws.append(W)
    b = 0.01 * np.ones(NUM_OUTPUT)
    bs.append(b)
    return np.hstack([W.flatten() for W in Ws] + [b.flatten() for b in bs])


def set_random_state(seed_num):
    np.random.seed(seed_num)  # Will be used by numpy and scipy.
    random.seed(seed_num)


def one_hot_encoder(y_data):
    num_label = int(np.max(y_data) + 1)
    return np.eye(num_label)[y_data]


def ReLU(x):
    return np.maximum(0, x)


def dReLU(x):
    return (x > 0).astype(int)


def computeAccuracy(yhat, y):
    return np.mean(np.argmax(y, axis=1) == np.argmax(yhat, axis=1))


def calcAccuracyAndLoss(x, y, weightsAndBiases):
    loss, _, _, y_hat = forward_prop(x, y, weightsAndBiases)
    acc = computeAccuracy(y_hat, y)
    return acc, loss


# %%
def compute_CE(y, y_hat):
    cross_entropy = (-1. / y.shape[0]) * np.sum(np.multiply(y, np.log(y_hat)))
    return np.sum(cross_entropy)


def softmax(z):
    denom = np.sum(np.exp(z), axis=1, keepdims=True)
    return np.exp(z) / denom


def forward_prop(X, y, weightsAndBiases):
    # print("Forward Prop")
    Ws, bs = unpack(weightsAndBiases)
    zs = []
    hs = []
    for layer in range(len(Ws) - 1):
        if layer == 0:
            z = np.dot(X, Ws[layer]) + bs[layer]
            zs.append(z)
        else:
            z = np.dot(hs[layer - 1], Ws[layer]) + bs[layer]
            zs.append(z)

        hs.append(ReLU(z))
    z = np.dot(hs[-1], Ws[-1]) + bs[-1]
    yhat = softmax(z)
    loss = (-1. / len(y)) * np.sum((y * np.log(yhat)))
    return loss, zs, hs, yhat


# %%
def back_prop(x, y, weightsAndBiases):
    # print("Inside Backprop")

    ## Unpacking the weights
    Ws, bs = unpack(weightsAndBiases)
    ## Getting loss, zs, hs and yhat from forward_prop
    loss, zs, hs, yhat = forward_prop(x, y, weightsAndBiases)
    # Cost function gradients with respect to model params
    dJdWs = []
    dJdbs = []
    dJdz = yhat - y
    dJdw = (1.0 / len(dJdz)) * np.dot(hs[-1].T, dJdz)
    dJdb = (1.0 / len(dJdz)) * np.sum(dJdz, axis=0)
    dJdz = (1.0 / len(dJdz)) * np.dot(dJdz, Ws[-1].T) * dReLU(zs[-1])
    dJdWs.append(dJdw)
    dJdbs.append(dJdb)
    for layer in range(NUM_HIDDEN_LAYERS - 1, 0, -1):
        z = zs[layer]
        h = hs[layer - 1]
        w = Ws[layer]
        b = bs[layer]

        dJdw = np.dot(h.T, dJdz)
        dJdb = np.sum(dJdz, axis=0)
        dJdWs.append(dJdw)
        dJdbs.append(dJdb)
        dJdz = np.dot(dJdz, w.T) * dReLU(zs[layer - 1])

        # For the last hidden layer:
    dJdw = np.dot(x.T, dJdz)
    dJdb = np.sum(dJdz, axis=0)
    dJdWs.append(dJdw)
    dJdbs.append(dJdb)
    dJdWs.reverse()
    dJdbs.reverse()

    ## Concatenate gradients
    return np.hstack([dJdW.flatten() for dJdW in dJdWs] + [dJdb.flatten() for dJdb in dJdbs])


# %%
def split_validation(features, labels, valid_ratio):
    valid_size = (int)(valid_ratio * features.shape[0] + 0.5)
    permuted_ind = np.random.permutation(features.shape[0])
    batch_list = []
    # append the current batched features and labels in a list
    batch_list.append((features[permuted_ind[0:valid_size]],
                       labels[permuted_ind[0:valid_size]]))
    batch_list.append((features[permuted_ind[valid_size:]],
                       labels[permuted_ind[valid_size:]]))
    return (batch_list)


# %%
def next_batch(features, labels, batch_size):
    batch_list = []
    for data in np.arange(0, np.shape(features)[0], batch_size):
        batch_list.append((features[data:data + batch_size], labels[data:data + batch_size]))
    return batch_list


# %%
def SGD(X, y, weightAndBiases, batch_size=10, epsilon=0.1, NUM_EPOCHS=int(10), alpha=0.01):
    weightsAndBiases = initWeightsAndBiases()
    Ws, Bs = unpack(weightsAndBiases)
    split_data = split_validation(X, y, 0.2)
    (validX, validY) = split_data[0]
    (trainX, trainY) = split_data[1]

    # Updating trajectory:
    trajectory = []
    cost_history = []
    # print("Epoch: ",NUM_EPOCHS)
    for epoch in range(NUM_EPOCHS):
        batch_loss = []
        WandB_list = []
        i = 0
        if epoch <= 10:
            epsilon_decay = epsilon
        elif epoch <= 20 and epoch > 10:
            epsilon_decay = epsilon * 0.5
        else:
            epsilon = 0.01
        start_epoch = time()
        for (batchX, batchY) in next_batch(trainX, trainY, batch_size):
            dWandB = back_prop(batchX, batchY, weightsAndBiases)
            weightsAndBiases -= epsilon_decay * dWandB
            acc_valid, cost = calcAccuracyAndLoss(validX, validY, weightsAndBiases)
            batch_loss.append(cost)
            WandB_list.extend(dWandB)

        # After all the batches:
        print("Time: ", time() - start_epoch)
        print("Cost: ", np.average(batch_loss))
        print("Accuracy: ", acc_valid)
        print("Epoch: ", epoch)
        trajectory.append(WandB_list)
        cost_history.append(np.average(batch_loss))
    return weightsAndBiases, trajectory, cost_history


# %%
# def train (trainX, trainY, weightsAndBiases, testX, testY):
#     NUM_EPOCHS = 100
#     trajectory = []
#     for epoch in range(NUM_EPOCHS):
#         # TODO: implement SGD.
#         # TODO: save the current set of weights and biases into trajectory; this is
#         # useful for visualizing the SGD trajectory.

#     return weightsAndBiases, trajectory
# %%
def plotSGDPath(trainX, trainY, trajectory):
    # TODO: change this toy plot to show a 2-d projection of the weight space
    # along with the associated loss (cross-entropy), plus a superimposed
    # trajectory across the landscape that was traversed using SGD. Use
    # sklearn.decomposition.PCA's fit_transform and inverse_transform methods.
    # Fitting PCA:
    pca = PCA(n_components=2)
    trajectory_pca = np.asarray(trajectory)
    Zs = pca.fit_transform(trajectory_pca)

    def toyFunction(x1, x2):
        WandB = pca.inverse_transform([x1, x2])
        loss, _, _, _ = forward_prop(trainX, trainY, WandB)
        return loss

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Compute the CE loss on a grid of points (corresonding to different w).
    axis1 = np.arange(-10, 10, 0.5)  # Just an example
    axis2 = np.arange(-10, 10, 0.5)  # Just an example
    Xaxis, Yaxis = np.meshgrid(axis1, axis2)
    Zaxis = np.zeros((len(axis1), len(axis2)))
    for i in range(len(axis1)):
        for j in range(len(axis2)):
            Zaxis[i, j] = toyFunction(Xaxis[i, j], Yaxis[i, j])
    ax.plot_surface(Xaxis, Yaxis, Zaxis, alpha=0.6)  # Keep alpha < 1 so we can see the scatter plot too.

    # Now superimpose a scatter plot showing the weights during SGD.
    Xaxis = np.linspace(-15, 15, num=10)
    Yaxis = np.linspace(-15, 15, num=10)
    _, _, Zaxis = SGD(trainX[:2500], trainY[:, 2500], weightsAndBiases)
    ax.scatter(Xaxis, Yaxis, Zaxis, color='r')
    ax.scatter(Xaxis, Yaxis, Zaxis, color='r')

    plt.show()


# %%
if __name__ == "__main__":
    # TODO: Load data and split into train, validation, test sets
    trainX = np.reshape(np.load("fashion_mnist_train_images.npy"), (-1, 784)) / 255.0
    trainY = np.load("fashion_mnist_train_labels.npy")
    testX = np.reshape(np.load("fashion_mnist_train_images.npy"), (-1, 784)) / 255.0
    testY = np.load("fashion_mnist_test_labels.npy")

    # Preprocessing:
    trainY = one_hot_encoder(trainY)
    testY = one_hot_encoder(testY)

    # Initialize weights and biases randomly
    weightsAndBiases = initWeightsAndBiases()
    print("Weights and Biases Shape: ", np.shape(weightsAndBiases))  # (8180, )
    # print(np.shape(weightsAndBiases[0]))    #
    # # Perform gradient check on random training examples
    print(scipy.optimize.check_grad(
        lambda wab: forward_prop(np.atleast_2d(trainX[0:5]), np.atleast_2d(trainY[0:5]), wab)[0], \
        lambda wab: back_prop(np.atleast_2d(trainX[0:5]), np.atleast_2d(trainY[0:5]), wab), \
        weightsAndBiases))
    start_time = time()
    epoch_list = np.array([5, 10, 15, 20])
    epsilon_list = np.array([0.5, 0.1, 0.05, 0.01, 0.001])
    weightsAndBiases_list = []
    trajectory_list = []
    cost_history_list = []
    for i in range(len(epoch_list)):
        for j in range(len(epsilon_list)):
            print('Current epoch - ' + str(epoch_list[i]) + ' and current epsilon - ' + str(epsilon_list[j]))
            weightsAndBiases, trajectory, cost_history = SGD(trainX, trainY, weightsAndBiases, batch_size=32,
                                                             epsilon=epsilon_list[j], NUM_EPOCHS=epoch_list[i],
                                                             alpha=0.01)
            weightsAndBiases_list.append(weightsAndBiases)
            trajectory_list.append(trajectory)
            cost_history_list.append(cost_history)
    print(np.shape(weightsAndBiases_list))
    print(np.shape(trajectory_list))
    print(np.shape(cost_history_list))
    weightsAndBiases, trajectory, cost_history = SGD(trainX, trainY, weightsAndBiases, batch_size=100, epsilon=0.2,
                                                     NUM_EPOCHS=2, alpha=0.01)

    Best_weightsAndBiases = []
    best_cost = None
    counter = 0
    for miniBatch in [32, 64]:
        for epsilon in [0.1, 0.2]:
            for epoch in [15, 20]:
                counter += 1
                print(counter, 'HyperParameter:',
                      'miniBatch:', miniBatch, 'epsilon:', epsilon, 'hidden layers:', NUM_HIDDEN_LAYERS, '# epochs:',
                      NUM_EPOCHS)
                # SGD method and grid search
                weightsAndBiases, trajectory, cost_history = SGD(trainX, trainY, weightsAndBiases, miniBatch, epsilon,
                                                                 epoch, alpha=0.01)
                if best_cost > cost_history:
                    Best_weightsAndBiases = weightsAndBiases
                    print("Best weight received!")
    loss_train, accu_train, loss_test, accu_test = CalcTestVsTrainAccuracy(trainX, trainY, testX, testY,
                                                                           Best_weightsAndBiases)
    print("Test Accuracy:", accu_test)
    print(time() - start_time)
    # # Plot the SGD trajectory
    plotSGDPath(trainX, trainY, trajectory)

# %%