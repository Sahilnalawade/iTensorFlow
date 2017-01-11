# iTensorFlow


this is a set of starter examples for linking R-based image processing
to keras/tensorflow based deep learning.

currently, the only useful contributions:

```
buildConstantHeaderSyntheticData.R
buildVaryingHeaderSyntheticData.R
```

to run one of the scripts and generate data, do:

```
cd iTensorFlow
# open R, then
source( "src/R/buildVaryingHeaderSyntheticData.R" )
```

this will generate a synthetic dataset of 100 2D images.  you can generate 3D
example datasets by modifying the script with `mydim=3`. each image is a noisy
sphere with a thin rim of lighter intensity embedded in a background of noise.
there are 50 images in each of 2 classes.  the 2nd class has a thicker rim.

this synthetic data allows us to test either regression or classification networks.
the regression problem has ground truth defined by the files:

```
# in iTensorFlow root dir
find ./data -name "*sv"
# ./data/dim2D/classification/spheres/spheres2CoM.csv
# ./data/dim2D/classification/varspheres/spheres2Transforms.csv
# ./data/dim3D/classification/spheres/spheres3CoM.csv
# ./data/dim2D/classification/varspheres/spheres3Transforms.csv
```

the interesting problem is `spheres*Transforms.csv` which may be used to perform
supervised learning of the physical space geometric mapping from the image data
to a fixed template coordinate system.  ideally, learning the mapping should be
independent of the image spacing (sampling rate).  Currently, the parameters to
learn include non-uniform stretching, rotations and translations.

the classification problem is just based on the thickness of the rim - just a
two class problem.

each image is randomly perturbed from the center position which gives some
learnable variability in the CoM.

there are also single and multichannel versions of each image.  the motivation
for including the spatial channels (in the multichannel version) is such that we
can pass the critical spatial information along with the raw intensity information.

there are many potential uses for this extra infomation that we can get into later
but, for now, it will be useful to demonstrate the learning with either single
or multichannel data.   in this simple case, performance should be identical.

in the `src/keras` directory, we have examples of building basic networks and
reading images.

there is currently one **missing link** between the synthetic R-based data and
the keras code:  we need to be able to read nifti images (via nibabel) into a
keras ready data structure.


## some notes on conv nets, mostly with keras

[keras 3d conv code](https://github.com/fchollet/keras/issues/4099) but i think this should work with tensorflow backend now

[3d cnn action example](http://learnandshare645.blogspot.com/2016/06/3d-cnn-in-keras-action-recognition.html)

[shapes 3d here](http://aetros.com/adrienj/3DCNN/code)

​copied to​ [here](http://aetros.com/stnava/3DCNN/code)

​maybe this would be a good one to try to reproduce:​

[C3D model used with a fork of Caffe to the Sports1M dataset migrated to Keras](https://imatge.upc.edu/web/resources/c3d-model-keras-trained-over-sports-1m)

[C3D model used with a fork of Caffe to the Sports1M dataset migrated to Keras gist](https://gist.github.com/albertomontesg/d8b21a179c1e6cca0480ebdf292c34d2)

[a u-net](https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py)

[kaggle tutorial for dicom heart data](https://github.com/jocicmarko/kaggle-dsb2-keras)

[auto encoder for image segmentation](http://pradyu1993.github.io/2016/03/08/segnet-post.html)

[brain lesion segmentation](https://github.com/naldeborgh7575/brain_segmentation)

[deep medic](https://github.com/Kamnitsask/deepmedic)


## a decent example from data creation to generalizable output

** very critical in below examples is to have correct alphanumeric ordering
of the written filenames **

prepare data from R

```
Rscript src/R/buildConstantHeaderSyntheticDataNumpy.R
```

pickle data in python.  one passes the image directory and csv for the train
data and the script will replace `train` with `test` in the directory and
file name and generate pickles for both train and test datasets.

```
# this function assumes input data is square and dimensions can be inferred
# from the sqrt of the array length (for 2D)
python src/r2python/pickleNpyData.py  -d 2 \
  -i data/dim2D/regression/spheresRad/train/singlechannel/ \
  -j data/dim2D/regression/spheresRad/train/singlechannel/spheres2Radius.csv
```

run mnist-based convnet

```
python src/keras/veryBasicKerasImageConvnet.py
```

## a segmentation example from data creation to generalizable output

WIP - not done

prepare data from R

```
Rscript src/R/buildConstantHeaderSyntheticSegmentationDataNumpy.R
```

pickle data in python.

```
python src/r2python/pickleNpyData.py -d 2 \
  -i data/dim2D/segmentation/spheresRad/train/singlechannel/ \
  -j data/dim2D/segmentation/spheresRad/train/singlechannel/spheres2Segmentation.csv
```

run mnist-based convnet

```
python src/keras/veryBasicKerasSegmentationConvnet.py
```
