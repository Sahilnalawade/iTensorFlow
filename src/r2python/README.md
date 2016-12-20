
1.  your suggested numpy format where:
  * this requires getting the array to vector mapping correct for both R to py and py to R
  * may be simple but requires validation + more work to standardize
  * proposed standardization:
    * a single-channel image of any dimensionality is a numpy vector 
    * a multi-channel image of any dimensionality is a numpy matrix
    * need to get this mapping consistent in py and R and test
    * python will need to know this additional data ( dimensionality, rows, etc ) to interpret successfully
    * this can be dealt with mostly in R to do the data reordering appropriate for numpy

2. simple itk i/o 
  * more flexible than nibabel in terms of i/o formats
  * consistent with antsr
  * seems to read directly to an array:

import numpy as np
from skimage import io
import SimpleITK as sitk
fn = '/tmp/kbykbykfloatmc.nii.gz'
myimg = io.imread( fn, plugin='simpleitk').astype(float)

  * simple translation from antsr : appears to be just the "by-slice" transpose of the data
