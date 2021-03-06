## standard imports
## http://datascience.stackexchange.com/questions/13428/what-is-the-significance-of-model-merging-in-keras
## merge prior (positional) information and image models
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
from keras.layers import Dense, Input, Flatten, Dropout, Activation, Reshape
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, SeparableConvolution2D
from keras.optimizers import SGD, RMSprop
from keras.regularizers import l2, activity_l2, l1, activity_l1
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.utils.np_utils import to_categorical

base_dir = os.environ.get('HOME')+'/code/iTensorFlow/'
img_fn = os.path.join( base_dir,'data/dim2D/segmentation/spheresRad/train/multichannel/Image_all.npz')
seg_fn = os.path.join( base_dir,'data/dim2D/segmentation/spheresRad/train/multichannel/Seg_all.npz')
teimg_fn = os.path.join( base_dir,'data/dim2D/segmentation/spheresRad/test/multichannel/Image_all.npz')
teseg_fn= os.path.join( base_dir,'data/dim2D/segmentation/spheresRad/test/multichannel/Seg_all.npz')

# read numpy data
X_train = np.load( img_fn )['arr_0']
Y_train = np.load( seg_fn )['arr_0']
Y_trainC = to_categorical( Y_train )
nclasses = Y_trainC.shape[1]
Y_train = Y_trainC.reshape( Y_train.shape[0], nclasses, Y_train.shape[1] * Y_train.shape[2] )

X_test = np.load( teimg_fn )['arr_0']
Y_test = np.load( teseg_fn )['arr_0']
Y_testC = to_categorical( Y_test )
Y_test = Y_testC.reshape( Y_test.shape[0], nclasses, Y_test.shape[1] * Y_test.shape[2] )

nc = X_test.shape[1]
nx = X_test.shape[2]

if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], nc, nx, nx)
    X_test = X_test.reshape(X_test.shape[0], nc, nx, nx)
    input_shape = ( nc, nx, nx )
else:
    X_train = X_train.reshape(X_train.shape[0], nx, nx, nc)
    X_test = X_test.reshape(X_test.shape[0], nx, nx, nc)
    input_shape = ( nx, nx, nc )

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples', 'groundTruth', Y_train.shape[0])
print(X_test.shape[0], 'test samples', 'groundTruth', Y_test.shape[0])


addPos = False
nb_filters = 32
kernel_size = (3, 3)
kernel_size2 = (5, 5)
pool_size = (2, 2)
l1r = 1.e-3

def getnet():
    nf=16
    inputs = Input((nc, nx, nx))
    conv1 = Convolution2D(nf, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Dropout( 0.25 )( conv1 )
    conv1 = Convolution2D(nf, 3, 3, activation='relu', border_mode='same')(conv1)
    conv2 = Convolution2D(nclasses, 1, 1, activation='sigmoid')(conv1)
    imgout = Flatten( )( conv2 )
    dx = Dense( nx * nx * nclasses , activation='softmax' )( imgout )
    dxr = Reshape( (nclasses, nx * nx) )( dx )
    model = Model(input=inputs, output=dxr )
    return model

umodel = getnet()
umodel.compile( optimizer='adamax', loss='categorical_crossentropy' )
batch_size = 200
nb_epoch = 25
umodel.fit( X_train, Y_train , batch_size=batch_size, nb_epoch=nb_epoch, verbose=2 )

# import pydot_ng as pydot
# from keras.utils.visualize_util import plot
# plot(model, to_file='model.png')

trscore = umodel.evaluate( X_train, Y_train, verbose=0)
print('Train score:', trscore )
tescore = umodel.evaluate( X_test, Y_test, verbose=0)
print('Test score:', tescore )

teY_pred = umodel.predict( X_test )
tepredicted_classes = teY_pred.argmax( axis = 1 )
Y_testClasses = Y_test.argmax( axis = 1 )
import pandas
ss = Y_testClasses.shape
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
print(classification_report( Y_testClasses.reshape( np.prod( ss ) ), tepredicted_classes.reshape( np.prod( ss ) ) ) )
print(confusion_matrix( Y_testClasses.reshape( np.prod( ss )), tepredicted_classes.reshape( np.prod( ss )) ) )
