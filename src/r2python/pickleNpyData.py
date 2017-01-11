#!/usr/bin/python
import sys, getopt
def main(argv):
   imageDir = ''
   csvFile = ''
   try:
      opts, args = getopt.getopt(argv,"hi:j:",["imageDir=","csv="])
   except getopt.GetoptError:
      print( 'test.py -i <imageDir> -j <csvFile>' )
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print('test.py -i <imageDir> -j <csvFile>')
         sys.exit()
      elif opt in ("-i", "--imageDir"):
         imageDir = arg
      elif opt in ("-j", "--csv"):
         csvFile = arg

   import numpy as np
   import matplotlib.pyplot as plt
   import pandas as pd
   import os
   import glob
   from PIL import Image
   from scipy.misc import toimage
   base_dir = os.environ.get('HOME')+'/code/iTensorFlow/'
   img_dir = os.path.join( base_dir,imageDir)
   com_path= os.path.join( base_dir,csvFile)
   targets = np.array( pd.read_csv(com_path) )
   allnpy = sorted( glob.glob( img_dir + "*npy" ) )
   # make an array for all this
   n = len( allnpy )
   exdata = np.load( allnpy[1] )
   nx = int( np.sqrt( exdata.shape[0]  ) )
   exarr  = exdata.reshape( [nx, nx ])
   myarr = np.ones( ( n, nx, nx ) )
   for i in range( len( allnpy ) ) :
       myarr[ i,:,:] = np.load( allnpy[i] ).reshape( [nx, nx ])

   np.savez( img_dir + "all.npz", myarr )
   imageDir2 = imageDir.replace("train", "test")
   csvFile2 = csvFile.replace("train", "test")
   img_dir = os.path.join( base_dir,imageDir2)
   com_path= os.path.join( base_dir,csvFile2)
   tetargets = np.array( pd.read_csv(com_path) )
   allnpy = sorted( glob.glob( img_dir + "*npy" ) )
   # make an array for all this
   n = len( allnpy )
   exdata = np.load( allnpy[1] )
   nx = int( np.sqrt( exdata.shape[0]  ) )
   exarr  = exdata.reshape( [nx, nx ])
   temyarr = np.ones( ( n, nx, nx ) )
   for i in range( len( allnpy ) ) :
       temyarr[ i,:,:] = np.load( allnpy[i] ).reshape( [nx, nx ])

   np.savez( img_dir + "all.npz", temyarr )

if __name__ == "__main__":
   main(sys.argv[1:])
