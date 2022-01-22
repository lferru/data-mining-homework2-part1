# Luis Ferrufino
# G#00997076
# HW#2 (Part 1)
# 3/11/20
# svmBuilder.py

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
import time

xTrain = np.load('./MNISTXtrain1.npy')
yTrain = np.load('./MNISTytrain1.npy')
xTest = np.load('./MNIST_X_test_1.npy')
yTest = np.load('./MNIST_y_test_1.npy')

xTrain = xTrain.reshape(60000, 28*28)
xTest = xTest.reshape(10000, 28*28)
xTrain = xTrain.astype('float32')
xTest = xTest.astype('float32')
xTrain /= 255
xTest /= 255
x = np.concatenate((xTrain, xTest))
y = np.concatenate((yTrain, yTest))

print('Building the svm...')

model = SVC(kernel='sigmoid')
model.fit(xTrain, yTrain)
from sklearn.externals import joblib
joblib.dump(model, './svmThirdPlace.pkl')
#score = model.score(xTest, yTest)
#print('The mean accuracy is ' + str(score))
'''
skf = StratifiedKFold(n_splits=10)
skf.get_n_splits(x, y)
score = 0
tempus = time.time()
#model = SVC(kernel='rbf')
print("bigTick")

for i, j in skf.split(x, y):
    xTra, xTes = x[i], x[j]
    yTra, yTes = y[i], y[j]
    model = SVC(kernel='poly', degree=4)
    model.fit(xTra, yTra)
    score = score + model.score(xTes, yTes)
    print('tick--' + str(time.time() - tempus))
    tempus = time.time()

print('aye,  the mean accuracy be ' + str(score / 10))
'''
