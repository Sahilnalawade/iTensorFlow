library( ANTsR )
n = 3
thdimg = makeImage( rep(n,3), rnorm( n^3 ) )
antsImageWrite( thdimg, '/tmp/kbykbykfloat.mhd' )


library( RcppCNPy )
thdarr = as.array( thdimg )
npySave( "/tmp/randmat.npy", thdarr[,,2] )
# this will fail to write an array but will write a numeric vector that can be reshaped
npySave( "/tmp/randmat3.npy", thdarr )
# but the above does not have the correct shape ...

# try nifti and nibable
antsImageWrite( thdimg, '/tmp/kbykbykfloat.nii.gz' )


# save a multi-channel image and see if python can read it (via nibabel)
mimg = mergeChannels( list( thdimg, thdimg ) )
antsImageWrite( mimg, '/tmp/kbykbykfloatmc.nii.gz' )


# try hdf5
library( rhdf5 )
fn = '/tmp/myData.h5'
h5createFile( fn )
h5write( thdarr, file = fn , 'thdarr')
h5ls( fn )
