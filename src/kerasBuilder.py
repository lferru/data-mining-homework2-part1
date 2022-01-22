# Luis Ferrufino
# G#00997076
# HW#2 (Part 1)
# 3/8/20
# kerasBuilder.py

import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model
from keras.utils import to_categorical

xTrain = np.load('./noisyXTrain.npy')
yTrain = np.load('./MNISTytrain1.npy')
xTest = np.load('./noisyXTest.npy')
yTest = np.load('./MNIST_y_test_1.npy')

xTrain = xTrain.reshape(60000, 28*28)
xTest = xTest.reshape(10000, 28*28)
xTrain = xTrain.astype('float32')
xTest = xTest.astype('float32')
xTrain /= 255
xTest /= 255
yTrain = to_categorical(yTrain, 10)
yTest = to_categorical(yTest, 10)

print('Building the ANN...')

activ = 'relu'
nodes = 300
model = Sequential()
model.add(Dense(175, activation='softsign', input_shape=(28*28,)))
model.add(Dense(nodes, activation=activ))
model.add(Dense(nodes, activation=activ))
model.add(Dense(nodes, activation=activ))
model.add(Dense(10, activation='softmax'))
'''note how the last layer is the output layer, and its number of nodes
    match that of the number of class labels'''

model.compile(optimizer='rmsprop', 
                loss='categorical_crossentropy',
                metrics=['accuracy'])

model.fit(xTrain, yTrain, epochs=5)

score = model.evaluate(xTest, yTest, verbose=0)
print('Loss in test: ', score[0])
print('Accuracy in test: ', score[1])

model.save('./noisyModel.h5')
