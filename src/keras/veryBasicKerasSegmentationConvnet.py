# very trivial example that shows the basics of data creation
# and building a simple neural network in keras - with train/test
## standard imports
import numpy as np
import matplotlib.pyplot as plt
import plotly.plotly as py
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
img_fn = os.path.join( base_dir,'data/dim2D/segmentation/spheresRad/train/singlechannel/all.npz')
com_path= os.path.join( base_dir,'data/dim2D/segmentation/spheresRad/train/singlechannel/spheres2Segmentation.csv')
teimg_fn = os.path.join( base_dir,'data/dim2D/segmentation/spheresRad/test/singlechannel/all.npz')
tecom_path= os.path.join( base_dir,'data/dim2D/segmentation/spheresRad/test/singlechannel/spheres2Segmentation.csv')

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
print(X_train.shape[0], 'train samples', 'groundTruth', Y_train.shape[0])
print(X_test.shape[0], 'test samples', 'groundTruth', Y_test.shape[0])

def unet_conv():
	img_rows = img_cols = nx
	inputs = Input((1, img_rows, img_cols))
	conv1 = Convolution2D(2, 3, 3, activation='relu', border_mode='same')(inputs)
	conv1 = Convolution2D(4, 3, 3, activation='relu', border_mode='same')(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
	conv2 = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(pool1)
	conv2 = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
	conv3 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(pool2)
	conv3 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
	conv4 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(pool3)
	conv4 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv4)
	pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
	conv5 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool4)
	conv5 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv5)
	up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
	conv6 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up6)
	conv6 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv6)
	up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
	conv7 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(up7)
	conv7 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(conv7)
	up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
	conv8 = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(up8)
	conv8 = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(conv8)
	up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
	conv9 = Convolution2D(4, 3, 3, activation='relu', border_mode='same')(up9)
	flat1 = Flatten()(conv9)
	conv10 = Dense( Y_test.shape[1] )(flat1)
	model = Model(input=inputs, output=conv10)
	return model

def mnist_conv():
    nb_filters = 8
    kernel_size = (3, 3)
    pool_size = (2, 2)
    model = Sequential()
    model.add(Convolution2D( nb_filters, kernel_size[0], kernel_size[1],
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

batch_size = 4
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

i = 2
x, y = Y_test[:,i],Y_pred[:,i]
# plt.scatter(x, y, alpha=.1, s=400)
# plt.show()
