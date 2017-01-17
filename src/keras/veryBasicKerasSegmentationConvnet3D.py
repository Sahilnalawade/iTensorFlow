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
from keras.layers import Input, merge, Convolution3D, MaxPooling3D, UpSampling3D
from keras.optimizers import SGD, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.utils.np_utils import to_categorical


base_dir = os.environ.get('HOME')+'/code/iTensorFlow/'
img_fn = os.path.join( base_dir,'data/dim3D/segmentation/spheresRad/train/singlechannel/all.npz')
com_path= os.path.join( base_dir,'data/dim3D/segmentation/spheresRad/train/singlechannel/spheres3Segmentation.csv')
teimg_fn = os.path.join( base_dir,'data/dim3D/segmentation/spheresRad/test/singlechannel/all.npz')
tecom_path= os.path.join( base_dir,'data/dim3D/segmentation/spheresRad/test/singlechannel/spheres3Segmentation.csv')

# read numpy data
X_train = np.load( img_fn )['arr_0']
Y_trainC = np.array( pd.read_csv(com_path) )
# Y_trainCcomp = X_train[:,5,5].round()
Y_trainC = Y_trainC[:,Y_trainC.shape[1]-1]
Y_train = to_categorical( Y_trainC )

X_test = np.load( teimg_fn )['arr_0']
Y_testC = np.array( pd.read_csv(tecom_path) )
# Y_testCcomp = X_test[:,5,5].round()
Y_testC = Y_testC[:,Y_testC.shape[1]-1]
Y_test = to_categorical( Y_testC )

# check an image
# toimage(X_train[8,:,:]).show()

nx = X_test.shape[1]

if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, nx, nx, nx)
    X_test = X_test.reshape(X_test.shape[0], 1, nx, nx, nx)
    input_shape = (1, nx, nx, nx )
else:
    X_train = X_train.reshape(X_train.shape[0], nx, nx, nx, 1)
    X_test = X_test.reshape(X_test.shape[0], nx, nx, nx, 1)
    input_shape = (nx, nx, nx, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# X_train /= 2.5
# X_test /= 2.5
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples', 'groundTruth', Y_train.shape[0])
print(X_test.shape[0], 'test samples', 'groundTruth', Y_test.shape[0])

def mnist_conv():
    nb_filters = 32
    kernel_size = (3, 3, 3)
    pool_size = (2, 2, 2)
    model = Sequential()
    model.add(Convolution3D( nb_filters, kernel_size[0], kernel_size[1], kernel_size[2],
                            border_mode='valid',
                            input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution3D(nb_filters, kernel_size[0], kernel_size[1], kernel_size[2]))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=pool_size))
    model.add(Dropout(0.05))
    model.add(Flatten())
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dropout(0.05))
    model.add(Dense( Y_test.shape[1] ))
    model.add(Activation('softmax'))
    return model

model = mnist_conv()

rms = RMSprop()
# model.compile( loss='mse', optimizer=rms, metrics=['mse'] )
model.compile(loss='categorical_crossentropy', optimizer='adam')

batch_size = 256
nb_epoch = 50
model.fit( X_train, Y_train, batch_size=batch_size,
    nb_epoch=nb_epoch, verbose=2 )

trscore = model.evaluate(X_train, Y_train, verbose=0)
print('Train score:', trscore )
tescore = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', tescore )

################################################################################
Y_pred = model.predict( X_train )
predicted_classes = model.predict_classes(X_train)
correct_indices = np.nonzero( predicted_classes == Y_trainC )
incorrect_indices = np.nonzero( predicted_classes != Y_trainC )

################################################################################
tepredicted_classes = model.predict_classes(X_test)
tecorrect_indices = np.nonzero(tepredicted_classes == Y_testC )[0]
teincorrect_indices = np.nonzero(tepredicted_classes != Y_testC )[0]

fracr = float( tecorrect_indices.shape[0]  ) / float( Y_testC.shape[0] )
fracw = float( teincorrect_indices.shape[0] ) / float( Y_testC.shape[0] )
print( " done " )
print( str( fracr * 100.0 ) + "% right " + str( fracw * 100.0 ) + "% wrong" )
