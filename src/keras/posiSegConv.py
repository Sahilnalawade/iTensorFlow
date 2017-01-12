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
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D
from keras.optimizers import SGD, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.utils.np_utils import to_categorical

base_dir = os.environ.get('HOME')+'/code/iTensorFlow/'
img_fn = os.path.join( base_dir,'data/dim2D/segmentation/spheresRad/train/singlechannel/all.npz')
com_path= os.path.join( base_dir,'data/dim2D/segmentation/spheresRad/train/singlechannel/spheres2Segmentation.csv')
teimg_fn = os.path.join( base_dir,'data/dim2D/segmentation/spheresRad/test/singlechannel/all.npz')
tecom_path= os.path.join( base_dir,'data/dim2D/segmentation/spheresRad/test/singlechannel/spheres2Segmentation.csv')

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
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples', 'groundTruth', Y_train.shape[0])
print(X_test.shape[0], 'test samples', 'groundTruth', Y_test.shape[0])

def inception():
    nb_filters = 32
    kernel_size = (3, 3)
    pool_size = (2, 2)
    input_img = Input( shape = input_shape )
    tower_1 = Convolution2D( nb_filters, 1, 1, border_mode='same', activation='relu')(input_img)
    tower_1 = Convolution2D( nb_filters, 3, 3, border_mode='same', activation='relu')(tower_1)
    tower_2 = Convolution2D( nb_filters, 1, 1, border_mode='same', activation='relu')(input_img)
    tower_2 = Convolution2D( nb_filters, 5, 5, border_mode='same', activation='relu')(tower_2)
    output = merge( [ tower_1, tower_2 ], mode = 'concat', concat_axis = 1 )

nb_filters = 32
kernel_size = (3, 3)
pool_size = (2, 2)
image_input = Input( shape=input_shape )
x = Convolution2D( nb_filters, kernel_size[0], kernel_size[1], activation='relu' )( image_input )
x = Convolution2D( nb_filters, kernel_size[0], kernel_size[1], activation='relu' )( x )
x = MaxPooling2D( pool_size )( x )
x = Dropout( 0.05 )( x )
out = Flatten( )( x )
position_input = Input(shape=(2,), name='aux_input')
x = merge([out, position_input], mode='concat')
x = Dense( 32, activation='relu'  )( x )
x = Dropout( 0.05 )( x )
main_output = Dense( Y_test.shape[1], activation='softmax', name='main_output' )( x )
model = Model(input=[ image_input, position_input], output=[main_output] )
model.compile( optimizer='adam', loss='categorical_crossentropy' )

batch_size = 256
nb_epoch = 50
model.fit( [X_train, X_trainPos],  [Y_train], batch_size=batch_size,
    nb_epoch=nb_epoch, verbose=2 )
#    , validation_data=([X_test,X_testPos], Y_test))

trscore = model.evaluate([X_train, X_trainPos], Y_train, verbose=0)
print('Train score:', trscore )
tescore = model.evaluate([X_test,X_testPos], Y_test, verbose=0)
print('Test score:', tescore )

################################################################################
Y_pred = model.predict( [X_train, X_trainPos] )
predicted_classes = Y_pred.argmax( axis = 1 )
correct_indices = np.nonzero( predicted_classes == Y_trainClasses )[0]
incorrect_indices = np.nonzero(predicted_classes != Y_trainClasses )[0]

################################################################################
teY_pred = model.predict( [X_test, X_testPos] )
tepredicted_classes = teY_pred.argmax( axis = 1 )
tecorrect_indices = np.nonzero( tepredicted_classes == Y_testClasses )[0]
teincorrect_indices = np.nonzero( tepredicted_classes != Y_testClasses )[0]

fracr = float( tecorrect_indices.shape[0]  ) / float( Y_testClasses.shape[0] )
fracw = float( teincorrect_indices.shape[0] ) / float( Y_testClasses.shape[0] )
print( " done " )
print( str( fracr * 100.0 ) + "% right " + str( fracw * 100.0 ) + "% wrong" )
