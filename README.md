# iTensorFlow


this is a set of starter examples for linking R-based image processing
to keras/tensorflow based deep learning.

currently, the only useful contribution here is:

```
buildSyntheticClassificationData.R
```

which will generate a synthetic dataset of 100 images.  each image is a noisy
sphere with a thin rim of lighter intensity embedded in a background of noise.
there are 50 images in each of 2 classes.  the 2nd class has a thicker rim.

this synthetic data allows us to test either regression or classification networks.
the regression problem has only one class with ground truth defined by the files:

```
# in iTensorFlow root dir
find ./data -name "*sv"
# ./data/dim2D/classification/spheres/spheres2CoM.csv
# ./data/dim3D/classification/spheres/spheres3CoM.csv
```

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
