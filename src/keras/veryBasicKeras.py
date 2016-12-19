# very trivial example that shows the basics of data creation
# and building a simple neural network in keras - with train/test
import numpy as numpy
# prepare numpy data
X_train = numpy.array([[1] * 128] * (10 ** 4) + [[0] * 128] * (10 ** 4))
X_test = numpy.array([[1] * 128] * (10 ** 2) + [[0] * 128] * (10 ** 2))

Y_train = numpy.array([True] * (10 ** 4) + [False] * (10 ** 4))
Y_test = numpy.array([True] * (10 ** 2) + [False] * (10 ** 2))

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

Y_train = Y_train.astype("bool")
Y_test = Y_test.astype("bool")

# build deep learning model
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
model = Sequential()
# takes a 128 vector as input and outputs a 50 node layer, densely connected
model.add(Dense(50, input_dim=128))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dropout(0.2))
# model.add(Dense(1, init='normal')) # for regression - just end here, no sigmoid layer
model.add(Dense(1)) # for classification
model.add(Activation('sigmoid')) # for classification, must add this

rms = RMSprop()
model.compile(loss='binary_crossentropy', optimizer=rms, metrics=['accuracy'])

batch_size = 32
nb_epoch = 3
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          show_accuracy=True, verbose=2, validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

zz = model.predict( X_test )
zz[2:5]
Y_test[2:5]
zz[102:105]
Y_test[102:105]
