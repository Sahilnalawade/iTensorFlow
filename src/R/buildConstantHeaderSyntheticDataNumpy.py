import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
from PIL import Image
from scipy.misc import toimage
base_dir = os.environ.get('HOME')+'/code/iTensorFlow/'
img_dir = os.path.join( base_dir,'data/dim2D/regression/spheresRad/train/singlechannel/')
com_path= os.path.join( base_dir,'data/dim2D/regression/spheresRad/train/singlechannel/spheres2Radius.csv')
targets = np.array( pd.read_csv(com_path) )
allnpy = glob.glob( img_dir + "*npy" )
# make an array for all this
n = len( allnpy )
exdata = np.load( allnpy[1] )
nx = int( np.sqrt( exdata.shape[0]  ) )
exarr  = exdata.reshape( [nx, nx ])
myarr = np.ones( ( n, nx, nx ) )
for i in range( len( allnpy ) ) :
    myarr[ i,:,:] = np.load( allnpy[i] ).reshape( [nx, nx ])

np.savez( img_dir + "all.npz", myarr )

img_dir = os.path.join( base_dir,'data/dim2D/regression/spheresRad/test/singlechannel/')
com_path= os.path.join( base_dir,'data/dim2D/regression/spheresRad/test/singlechannel/spheres2Radius.csv')
tetargets = np.array( pd.read_csv(com_path) )
allnpy = glob.glob( img_dir + "*npy" )
# make an array for all this
n = len( allnpy )
exdata = np.load( allnpy[1] )
nx = int( np.sqrt( exdata.shape[0]  ) )
exarr  = exdata.reshape( [nx, nx ])
temyarr = np.ones( ( n, nx, nx ) )
for i in range( len( allnpy ) ) :
    temyarr[ i,:,:] = np.load( allnpy[i] ).reshape( [nx, nx ])

np.savez( img_dir + "all.npz", temyarr )
