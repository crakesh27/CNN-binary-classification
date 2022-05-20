from keras.layers import Conv1D, Dense, MaxPool1D, Flatten
from keras.models import Sequential
from keras.layers.convolutional import Conv1D
from sklearn import preprocessing
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Part 1 - Data Preprocessing
# Importing the libraries
def accuracy_model(Y_test, Y_pred):
    correct = 0
    wrong = 0
    for i in range(len(Y_test)):
        if(Y_test[i] == Y_pred[i]):
            correct += 1
        else:
            wrong += 1
    return correct/(correct+wrong)

def plot(x, y):
    plt.plot(x, y)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Plot for cost vs iterations')
    plt.savefig('cnn_cost_iter_plt.png')
    plt.close()


# Importing the dataset
mat = scipy.io.loadmat('data_for_cnn.mat')
dataset = mat['ecg_in_window']
class_label = scipy.io.loadmat('class_label.mat')
labels = class_label['label']

normal = preprocessing.StandardScaler()
dataset = normal.fit_transform(dataset)


X_test = dataset[:100]
X_test = np.concatenate((X_test, dataset[-100:]))
X_test = dataset
Y_test = labels[:100]
Y_test = np.concatenate((Y_test, labels[-100:]))
Y_test = labels
X_train = dataset[100:900]
Y_train = labels[100:900]

X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)

# Part 2 - Now let's make the CNN!
num_epoch = 40
# Initialising the ANN
model = Sequential()
model.add(Conv1D(32, 10, activation='relu', input_shape=(1000, 1)))
model.add(MaxPool1D(pool_size=2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
# Fitting the ANN to the Training set
hist = model.fit(X_train, Y_train, batch_size=100, epochs=num_epoch)

iterations = np.arange(num_epoch) + 1
plot(iterations, hist.history['loss'])

# Part 3 - Making predictions and evaluating the model
# Predicting the Test set results
Y_pred = model.predict(X_test)
Y_pred = (Y_pred > 0.5)
print("Accuracy : ", end='')
print(accuracy_model(Y_test, Y_pred))
