# very trivial example that shows the basics of data creation
# and building a simple neural network in keras - with train/test
## standard imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import nibabel as nib
from PIL import Image
from scipy.misc import toimage
from keras.preprocessing import image
from keras.models import Model, Sequential
from keras.layers import merge, Dense, Input, Flatten, Dropout, Activation
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D
from keras.optimizers import SGD, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K


base_dir = os.environ.get('HOME')+'/code/iTensorFlow/'
img_fn = os.path.join( base_dir,'data/dim2D/regression/spheresRad/train/singlechannel/all.npz')
com_path= os.path.join( base_dir,'data/dim2D/regression/spheresRad/train/singlechannel/spheres2Radius.csv')
teimg_fn = os.path.join( base_dir,'data/dim2D/regression/spheresRad/test/singlechannel/all.npz')
tecom_path= os.path.join( base_dir,'data/dim2D/regression/spheresRad/test/singlechannel/spheres2Radius.csv')

# read numpy data
X_train = np.load( img_fn )['arr_0']
Y_train = np.array( pd.read_csv(com_path) )

X_test = np.load( teimg_fn )['arr_0']
Y_test = np.array( pd.read_csv(tecom_path) )

nx = X_test.shape[1]

if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, nx, nx)
    X_test = X_test.reshape(X_test.shape[0], 1, nx, nx)
    input_shape = (1, nx, nx)
else:
    X_train = X_train.reshape(X_train.shape[0], nx, nx, 1)
    X_test = X_test.reshape(X_test.shape[0], nx, nx, 1)
    input_shape = (nx, nx, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 2.5
X_test /= 2.5
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

def mnist_conv():
    nb_filters = 32
    kernel_size = (3, 3)
    pool_size = (2, 2)
    model = Sequential()
    model.add(Convolution2D( 32, kernel_size[0], kernel_size[1],
                            border_mode='valid', init = 'normal',
                            input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.05))
    model.add(Dense( Y_test.shape[1] ))
    return model

model = mnist_conv()

rms = RMSprop()
model.compile( loss='mse', optimizer=rms, metrics=['mse'] )

batch_size = 32
nb_epoch = 50
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=2, validation_data=(X_test, Y_test))

trscore = model.evaluate(X_train, Y_train, verbose=0)
print('Train score:', trscore[0])
print('Train accuracy:', trscore[1])
tescore = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', tescore[0])
print('Test accuracy:', tescore[1])

Y_pred = model.predict( X_train )
for i in range(Y_test.shape[1]):
	print( np.corrcoef(Y_train[:,i],Y_pred[:,i])[0,1] )

Y_pred = model.predict( X_test )
for i in range(Y_test.shape[1]):
	print( np.corrcoef(Y_test[:,i],Y_pred[:,i])[0,1] )

print("not bad")
