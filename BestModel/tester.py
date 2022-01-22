# Luis Ferrufino
# G#00997076
# 3/15/20
# tester.py
# HW2 (Part 1)

import keras
import numpy as np
from keras.models import load_model

model = load_model('./MyBestModel_g00997076.h5')
#xTest = np.load('../src/MNIST_X_test_1.npy')
#xTest = xTest.reshape(10000, 28*28)
xTest = np.load('./Secret_Test.npy')
xTest = xTest.reshape(xTest.size // (28*28), 28*28)
xTest = xTest.astype('float32')
xTest /= 255
predictions = model.predict_classes(xTest, verbose=0)
f = open('./results.txt', 'w+')

for i in range(0, predictions.size):

  f.write(str(predictions[i]))

  if i < predictions.size - 1:

      f.write('\n')
f.close()
