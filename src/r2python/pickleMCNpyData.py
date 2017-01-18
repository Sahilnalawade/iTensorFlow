#!/usr/bin/python
import sys, getopt
def main(argv):
   imageDir = ''
   myExt = ''
   try:
      opts, args = getopt.getopt(argv,"hi:j:d:",["imageDir=","ext=","dim="])
   except getopt.GetoptError:
      print( 'test.py -i <imageDir> -j extension -d <2>' )
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print('test.py -i <imageDir> -j extension -d <2>')
         sys.exit()
      elif opt in ("-i", "--imageDir"):
         imageDir = arg
      elif opt in ("-j", "--ext"):
         myExt = arg
      elif opt in ("-d", "--dim"):
         mydim = arg

   import numpy as np
   import matplotlib.pyplot as plt
   import pandas as pd
   import os
   import glob
   from PIL import Image
   from scipy.misc import toimage
   mydimf = float( mydim )
   print("pickle train data")
   base_dir = os.environ.get('HOME')+'/code/iTensorFlow/'
   img_dir = os.path.join( base_dir,imageDir)
#   com_path= os.path.join( base_dir,csvFile)
#   targets = np.array( pd.read_csv(com_path) )
   ext = "*"+myExt+"*.npy"
   allnpy = sorted( glob.glob( img_dir + ext ) )
   # make an array for all this
   n = int( len( allnpy ) )
   print( "have " + str(n) + " data points with ext " + ext )
   exdata = np.load( allnpy[1] )
   nx = int( round( exdata.shape[1] ** (1. / mydimf ) ) )
   nchan = int( exdata.shape[0] )
   if round( mydimf ) == 2:
     exarr  = exdata.reshape( [nchan, nx, nx ])
     myarr = np.ones( ( n, nchan, nx, nx ) )
     for i in range( len( allnpy ) ) :
        myarr[ i,:,:,:] = np.load( allnpy[i] ).reshape( [nchan, nx, nx ])
   if round( mydimf ) == 3:
     exarr  = exdata.reshape( [nchan, nx, nx, nx ])
     myarr = np.ones( ( n, nchan, nx, nx, nx ) )
     for i in range( len( allnpy ) ) :
        myarr[ i,:,:,:,:] = np.load( allnpy[i] ).reshape( [nchan, nx, nx, nx ])
   ofn = img_dir + myExt + "_all.npz"
   print( "write " + ofn )
   np.savez( ofn, myarr )
 # done
   imageDir2 = imageDir.replace("train", "test")
   img_dir = os.path.join( base_dir,imageDir2)
#   com_path= os.path.join( base_dir,csvFile2)
#   tetargets = np.array( pd.read_csv(com_path) )
   allnpy = sorted( glob.glob( img_dir + ext ) )
   # make an array for all this
   n = int( len( allnpy ) )
   print( "pickle test data:" + str( n ) )
   if round( mydimf ) == 2:
     exarr  = exdata.reshape( [nchan, nx, nx ])
     myarr = np.ones( ( n, nchan, nx, nx ) )
     for i in range( len( allnpy ) ) :
        myarr[ i,:,:,:] = np.load( allnpy[i] ).reshape( [nchan, nx, nx ])
   if round( mydimf ) == 3:
     exarr  = exdata.reshape( [nchan, nx, nx, nx ])
     myarr = np.ones( ( n, nchan, nx, nx, nx ) )
     for i in range( len( allnpy ) ) :
        myarr[ i,:,:,:,:] = np.load( allnpy[i] ).reshape( [nchan, nx, nx, nx ])
   ofn = img_dir + myExt + "_all.npz"
   print( "write " + ofn )
   np.savez( ofn, myarr )

if __name__ == "__main__":
   main(sys.argv[1:])
