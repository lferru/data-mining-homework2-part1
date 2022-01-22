# Luis Ferrufino
# G#00997076
# 3/14/20
# performance.py
# HW 2 (Part 1)

import keras
import numpy as np
#from sklearn.externals import joblib
from keras.models import load_model

model = load_model('./noisyModel.h5')
#model = joblib.load('./svmThirdPlace.pkl')
xTest = np.load('./noisyXTest.npy')
xTest = xTest.reshape(10000, 28*28)
xTest = xTest.astype('float32')
xTest /= 255
yTest = np.load('./MNIST_y_test_1.npy')
predictions = model.predict_classes(xTest, verbose=0)
#predictions = model.predict(xTest)
confusion = np.zeros((10,4), dtype=int)
accu = 0
f1 = 0

for i in range(0, predictions.size): #go through each prediction

    for j in range(0, 10): #go through each row of the confusion array

        if ( ( predictions[i] == j ) and ( yTest[i] == j ) ):

            confusion[j][0] += 1 # a true positive
        elif ( ( predictions[i] == j ) and ( yTest[i] != j ) ):

            confusion[j][1] += 1 # a false positive
        elif ( ( predictions[i] != j ) and ( yTest[i] == j ) ):

            confusion[j][2] += 1 # a false negative
        else:

            confusion[j][3] += 1 # a true negative

for i in range(0, 10): #go through each row of the confusion array
    
    avgAccu = (confusion[i][0] + confusion[i][3])
    avgAccu /= (confusion[i][0] + confusion[i][1] + confusion[i][2] + confusion[i][3])
    accu += avgAccu
    f1 += 2 * confusion[i][0] / (2 * confusion[i][0] + confusion[i][1] + confusion[i][2])
print("Average accuracy is " + str(accu / 10))
print("Average F1 Score is " + str(f1 / 10))
