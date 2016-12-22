import numpy as np
n = 3
filename = '/tmp/kbykbykfloat.raw'
with open(filename, 'rb') as f:
  data = np.fromfile(f, dtype=np.float32)
  array = np.reshape(data, [n, n, n])

# this array is the "by slice" transpose of the R result ...

# now try to read a RcppCNPy object
m = np.load("/tmp/randmat.npy")
m = np.reshape( np.load("/tmp/randmat3.npy"), [n, n, n])

# now a nifti multi-channel with nibabel
# see http://nipy.org/nibabel/gettingstarted.html
import os
import numpy as np
import nibabel as nib
img = nib.load( '/tmp/kbykbykfloat.nii.gz' )
data = img.get_data()
mimg = nib.load( '/tmp/kbykbykfloatmc.nii.gz' )
mata = mimg.get_data()


# now h5
from __future__ import print_function
import numpy as np
import h5py
fn = '/tmp/myData.h5'
with h5py.File( fn, 'r' ) as hf:
    print('List of arrays in this file: \n', hf.keys())
    data = hf.get('thdarr')
    np_data = np.array(data)
    print('Shape of the array dataset_1: \n', np_data.shape)


# now try the multi-channel image read with simpleitk
from skimage import io
import SimpleITK as sitk
fn = '/tmp/kbykbykfloatmc.nii.gz'
myimg = io.imread( fn, plugin='simpleitk').astype(float)


import numpy as np
n = 256
with open( fn, 'rb') as f:
   data = np.fromfile( f, dtype=np.double )
   array = np.reshape( data, [n, n] )


from PIL import Image
import numpy as np

w, h = 512, 512
data = np.zeros((h, w, 3), dtype=np.uint8)
data[256, 256] = [255, 0, 0]
img = Image.fromarray(data, 'RGB')
img.show()

w, h = 512, 512
data = np.zeros((h, w), dtype='d')
data[256, 256] = 1.5
img = Image.fromarray(data, 'F')
img.show()

from PIL import Image
import numpy as np
from scipy.misc import toimage
o = 182
m = 218
n = 182
data = np.load( ofn )
array = np.reshape( data, [ o, m, n ] )
toimage(array).show()
