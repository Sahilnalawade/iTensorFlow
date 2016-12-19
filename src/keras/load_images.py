"""
To read in images from disk, they must be in the following format:

main_dir/

	class1/

		img1.jpg
		img2.jpg
		img3.jpg

	class2/
		img4.jpg
		img5.jpg

It doesn't matter what these files/folders are called. There must be a
second folder after main_dir.

Here is the class, and all that it can do:

keras.preprocessing.image.ImageDataGenerator(featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=0.,
    width_shift_range=0.,
    height_shift_range=0.,
    shear_range=0.,
    zoom_range=0.,
    channel_shift_range=0.,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=None,
    dim_ordering=K.image_dim_ordering())
"""


import keras

# this is the class which does the work loading the images
# see keras/preprocessing/image.py for more
from keras.preprocessing.image import ImageDataGenerator


data_gen = ImageDataGenerator()

data_flow = data_gen.flow_from_directory(
						'rjpg',
						target_size=(150, 150),
						batch_size=32,
						class_mode=None)


# generate samples using data_flow.next()
x = data_flow.next()
print( x.shape )
