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
from keras.layers import Dense, Input, Flatten, Dropout, Activation
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, SeparableConvolution2D
from keras.optimizers import SGD, RMSprop
from keras.regularizers import l2, activity_l2, l1, activity_l1
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.utils.np_utils import to_categorical

base_dir = os.environ.get('HOME')+'/code/iTensorFlow/'
img_fn = os.path.join( base_dir,'data/dim2D/segmentation/spheresRad/train/multichannel/all.npz')
com_path= os.path.join( base_dir,'data/dim2D/segmentation/spheresRad/train/multichannel/spheres2Segmentation.csv')
teimg_fn = os.path.join( base_dir,'data/dim2D/segmentation/spheresRad/test/multichannel/all.npz')
tecom_path= os.path.join( base_dir,'data/dim2D/segmentation/spheresRad/test/multichannel/spheres2Segmentation.csv')

# read numpy data
X_train = np.load( img_fn )['arr_0']
Y_trainClassAndPos = np.array( pd.read_csv(com_path) )
Y_trainClasses = Y_trainClassAndPos[:,Y_trainClassAndPos.shape[1]-1]
X_trainPos = Y_trainClassAndPos[:,0:(Y_trainClassAndPos.shape[1]-1)]
Y_train = to_categorical( Y_trainClasses )

X_test = np.load( teimg_fn )['arr_0']
Y_testClassAndPos = np.array( pd.read_csv(tecom_path) )
Y_testClasses = Y_testClassAndPos[:,Y_testClassAndPos.shape[1]-1]
X_testPos = Y_testClassAndPos[:,0:(Y_testClassAndPos.shape[1]-1)]
Y_test = to_categorical( Y_testClasses )

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
image_input = Input( shape=input_shape, name='image_input' )
x = Convolution2D( nb_filters, kernel_size[0], kernel_size[1], activation='relu', W_regularizer=l1(l1r) )( image_input )
# x = SeparableConvolution2D( nb_filters, kernel_size[0], kernel_size[1], activation='relu' )( image_input )
# x = SeparableConvolution2D( nb_filters, kernel_size[0], kernel_size[1], activation='relu', activity_regularizer=l1(l1r) )( x )
x = MaxPooling2D( pool_size )( x )
x = Dropout( 0.25 )( x )
x = Flatten( )( x )
position_input = Input(shape=(2,), name='position_input')
# y = Dense( 12, W_regularizer=l1(l1r) )( position_input )
if addPos:
    x = merge( [ x, position_input ], mode='concat' )
x = Dense( 256, W_regularizer=l1(l1r), activation='relu'  )( x )
# x = Dropout( 0.25 )( x )
x = Dense( 256, W_regularizer=l1(l1r), activation='relu'  )( x )
x = Dropout( 0.25 )( x )
main_output = Dense( Y_test.shape[1], activation='softmax', name='Segmentation' )( x )
if addPos:
    model = Model( input=[ image_input, position_input], output=[main_output] )
else:
    model = Model( input=image_input, output=[main_output] )
model.compile( optimizer='adam', loss='categorical_crossentropy' )

# import pydot_ng as pydot
# from keras.utils.visualize_util import plot
# plot(model, to_file='model.png')

batch_size = 1000
nb_epoch = 100
if addPos == True:
   myinput = [ X_train, X_trainPos ]
   teinput = [ X_test, X_testPos ]
else:
   myinput = X_train
   teinput = X_test

model.fit( myinput,  [Y_train], batch_size=batch_size,
    nb_epoch=nb_epoch, verbose=2 )

trscore = model.evaluate( myinput, Y_train, verbose=0)
print('Train score:', trscore )
tescore = model.evaluate( teinput, Y_test, verbose=0)
print('Test score:', tescore )

################################################################################
Y_pred = model.predict( myinput )
predicted_classes = Y_pred.argmax( axis = 1 )
correct_indices = np.nonzero( predicted_classes == Y_trainClasses )[0]
incorrect_indices = np.nonzero(predicted_classes != Y_trainClasses )[0]
fracr = float( correct_indices.shape[0]  ) / float( Y_trainClasses.shape[0] )
fracw = float( incorrect_indices.shape[0] ) / float( Y_trainClasses.shape[0] )
print( " done " )
print( str( fracr * 100.0 ) + "% right " + str( fracw * 100.0 ) + "% wrong" )

################################################################################
teY_pred = model.predict( teinput )
tepredicted_classes = teY_pred.argmax( axis = 1 )
tecorrect_indices = np.nonzero( tepredicted_classes == Y_testClasses )[0]
teincorrect_indices = np.nonzero( tepredicted_classes != Y_testClasses )[0]

fracr = float( tecorrect_indices.shape[0]  ) / float( Y_testClasses.shape[0] )
fracw = float( teincorrect_indices.shape[0] ) / float( Y_testClasses.shape[0] )
print( " done " )
print( str( fracr * 100.0 ) + "% right " + str( fracw * 100.0 ) + "% wrong" )

import pandas
my_series = pandas.Series( Y_testClasses  )
print( "Test" )
print(  my_series.value_counts() )
my_series = pandas.Series( tepredicted_classes  )
print( "Test" )
print(  my_series.value_counts() )

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
# print( confusion_matrix( Y_testClasses, tepredicted_classes ) )
print(classification_report( Y_testClasses, tepredicted_classes ) )

# visualize some of the weights
kk = model.layers[1].get_weights()
W = np.squeeze(kk[0])
# toimage(W[0,:,:]).show() # first filter
# toimage(W[2,:,:]).show() # third filter

import numpy.ma as ma
import pylab as pl
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

def nice_imshow(ax, data, vmin=None, vmax=None, cmap=None):
    """Wrapper around pl.imshow"""
    if cmap is None:
        cmap = cm.jet
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax.imshow(data, vmin=vmin, vmax=vmax, interpolation='nearest', cmap=cmap)
    pl.colorbar(im, cax=cax)

def make_mosaic(imgs, nrows, ncols, border=1):
    """
    Given a set of images with all the same shape, makes a
    mosaic with nrows and ncols
    """
    nimgs = imgs.shape[0]
    imshape = imgs.shape[1:]

    mosaic = ma.masked_all((nrows * imshape[0] + (nrows - 1) * border,
                            ncols * imshape[1] + (ncols - 1) * border),
                            dtype=np.float32)

    paddedh = imshape[0] + border
    paddedw = imshape[1] + border
    for i in range(nimgs):
        row = int(np.floor(i / ncols))
        col = i % ncols

        mosaic[row * paddedh:row * paddedh + imshape[0],
               col * paddedw:col * paddedw + imshape[1]] = imgs[i]
    return mosaic

#pl.imshow(make_mosaic(np.random.random((9, 10, 10)), 3, 3, border=1))

pl.figure( figsize=( 15, 15 ) )
pl.title( 'conv1 weights' )
# nice_imshow( pl.gca(), make_mosaic(W, 6, 6), -0.1, 0.1, cmap=cm.binary)
