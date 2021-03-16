
# %%
import numpy as np
from tensorflow import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Dense, Flatten, ReLU
from sklearn.model_selection import train_test_split

# %% For Q1:
from tensorflow.keras.callbacks import ModelCheckpoint
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

## Loading Dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

## Preprocessing
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
(x_train, x_valid) = x_train[5000:], x_train[:5000]
(y_train, y_valid) = y_train[5000:], y_train[:5000]

img_width, img_height = 28, 28
x_train = x_train.reshape(x_train.shape[0], img_width, img_height, 1)
x_valid = x_valid.reshape(x_valid.shape[0], img_width, img_height, 1)
x_test = x_test.reshape(x_test.shape[0], img_width, img_height, 1)
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_valid = tf.keras.utils.to_categorical(y_valid, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

## Q1:
model = tf.keras.Sequential()
# Must define the input shape in the first layer of the neural network
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28 ,28 ,1)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
# Separate ReLU? or within the layer?
# Take a look at the model summary
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5', verbose = 1, save_best_only=True)
model.fit(x_train,
          y_train,
          batch_size=64,
          epochs=100,
          validation_data=(x_valid, y_valid),
          callbacks=[checkpointer])
# Load the weights with the best validation accuracy
model.load_weights('model.weights.best.hdf5')

# Evaluate the model on test set
score = model.evaluate(x_test, y_test, verbose=0)

# Print test accuracy
print('\n', 'Test accuracy:', score[1])

# %%
## For Q2:
model = tf.keras.Sequential()
model.add(Conv2D(filters = 64, kernel_size = 3, padding = 'valid', input_shape = (28, 28, 1)))
model.add(MaxPool2D(pool_size = 2, strides = 2, padding = 'valid'))
model.add(ReLU())
model.add(Flatten())
model.add(Dense(1024))
model.add(ReLU())
model.add(Dense(10, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

## Q2 - Custom Convolution.

# model = keras.models.load_model("model_weight.h5")
img_dim = (img_height, img_width)
my_callbacks =  tf.keras.callbacks.ModelCheckpoint(filepath='model_weight.h5', verbose = 1, save_best_only=True)
history_custom = model.fit(x_train, y_train, batch_size = 64, epochs = 2, validation_data = (x_valid, y_valid), callbacks = my_callbacks)
test_score = model.evaluate(x_test, y_test, verbose = 0)
print("Test Score: ", test_score[1])
# %%
def ffnn_converter(model):
    """
    Extract Weights and Biases from a given model
    """
    _, _, W2_Dense, b2_Dense, W3_Dense, b3_Dense = list(map(lambda var: var.numpy(), model.trainable_variables))
    W1_conv, b1_conv = model.layers[0].get_weights()
    num_filters = W1_conv.shape[-1] ## last index gives us num of filters!
    # Shape of Layer: (3, 3, 1, 64)

    output_dim = (img_height - W1_conv.shape[0 ] +1) * (img_width - W1_conv.shape[1] + 1)
    b1_conv = np.transpose(np.repeat(np.reshape(b1_conv, (1, -1)), output_dim, axis = 0))

    ## For all the 64 Filters:
    conv_filters = []
    for index in range(num_filters):
        conv_filter = W1_conv[:, :, 0, index]
        conv_filters.append(convert_weights(conv_filter, (img_height, img_width)))
    W1_conv = np.array(conv_filters)
    return W1_conv, b1_conv, W2_Dense, b2_Dense, W3_Dense, b3_Dense

def convert_weights(conv_filter, img_dim):
    """
    Given a convolution filter, convert it into a vector similar to that of a FFNN layer.
    """
    rows = img_dim[0]
    cols = img_dim[1]
    num_rows, num_cols = len(conv_filter), len(conv_filter)
    weight_dim = ((rows - num_rows + 1) * (cols - num_cols + 1), (rows * cols))

    # The number of zeros to fill up for every row
    row_zero = cols - len(conv_filter[0])
    fill_zeros = np.zeros(row_zero)
    flat_filter = []
    for row in conv_filter[:-1]:
        flat_filter.extend(list(row))
        flat_filter.extend(list(fill_zeros))
    flat_filter.extend(list(conv_filter[-1]))
    flat_filter = np.array(flat_filter)

    filter_transform = np.zeros(weight_dim)
    # This is done to create a matrix that mimics the convolusion operation
    for row in range(len(filter_transform)):
        start_id = (row % (cols - num_col s +1)) + ((ro w/ /(cols - num_cols + 1)) * cols)
        end_id = start_id + len(flat_filter)
        filter_transform[row][start_id: end_id] = flat_filter
    return filter_transform

def ff_layer(W, b, X):
    return W.T @ X + b

def MaxPool(x, pool_size):
    """
    Perform MaxPooling for a feedforward Layer!
    """
    pool_dim = (img_width - pool_size, img_height - pool_size)
    pooled_map = []
    for map_x in x:
        reshaped_map = np.reshape(map_x, pool_dim)
        pooled = []
        for row in range(0, pool_dim[0], pool_size):
            row_pool = []
            for col in range(0, pool_dim[1], pool_size):
                maxVal = -np.inf

                # For the 2 x 2 feature map considered!
                for ff_row  in range(pool_size):
                    for ff_col in range(pool_size):
                        maxVal = max(maxVal, reshaped_map[row + ff_row][col + ff_col])
                row_pool.append(maxVal)
            pooled.append(np.array(row_pool))
        pooled_map.append(np.array(pooled).flatten())
    return np.array(pooled_map)


def softmax(x):
    exp_x = np.exp(x)
    sum_ex = np.sum(exp_x)
    return exp_x / sum_ex

def relu(x):
    return np.maximum(0, x)


# W1_conv, b1_conv, W2_dense, b2_dense, W3_dense, b3_dense
W1_conv, b1_conv, W2_dense, b2_dense, W3_dense, b3_dense = ffnn_converter(model)
sample_custom = x_train[:1 ,: ,: ,:].flatten()
conv_output = ff_layer(np.transpose(W1_conv), b1_conv, sample_custom)
maxPool_output = MaxPool(conv_output, 2)
act_output = relu(np.transpose(maxPool_output).flatten())
dense1_output = ff_layer(W2_dense, b2_dense, act_output)
dense1_relu = relu(dense1_output)
dense2_output = ff_layer(W3_dense, b3_dense, dense1_relu)
softmax_output = softmax(dense2_output)


# %%
print("Converted Prediction: \n", softmax_output)
print("Tf Prediction: \n", model.predict(x_train[:1 ,: ,: ,:])[0])

