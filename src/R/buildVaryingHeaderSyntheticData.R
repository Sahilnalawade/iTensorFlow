library( ANTsR )
library( jpeg )
library( ANTsRNpy )
ext2 = ".npy"
# run this in iTensorFlow base directory
temp = unlist(strsplit(getwd(),"/"))
if ( temp[ length( temp ) ] != "iTensorFlow" )
  stop("run this script within the iTensorFlow base dir")
n = 50
if ( ! exists("mydim") ) mydim = 2
if ( mydim == 2 ) ext = ".nii.gz" else ext = ".nii.gz"
idim = rep( 64, mydim )
if ( mydim == 3 ) idim = rep( 32, mydim ) # something smaller for 3D testing/speed
odir = paste( "data/dim", length(idim), "D/classification/varspheres", sep='' )
baserad = 15
plusrad = c( 3, 5 )
classes = c("class1","class2")
if ( mydim == 3 ) {
  comdf = data.frame( x=0, y=0, z=0 )
  txdf = data.frame( matrix( nrow=1, ncol=10 ) )
  colnames( txdf ) = paste( "p", 1:ncol(txdf), sep='' )
}
if ( mydim == 2 ) {
  comdf = data.frame( x=0, y=0 )
  txdf = data.frame( matrix( nrow=1, ncol=6 ) )
  colnames( txdf ) = paste( "p", 1:ncol(txdf), sep='' )
}
myct = 1
useconstspacing = FALSE
for ( ct in 1:2 ) {
  for ( k in 1000:(1000+n-1) ) {
# this is where we make images with constant voxel dimensions but with
# different header information.  this spatial information will be passed
# as extra channels.  note that these channels, generally speaking, encode
# **spatial correspondence** which is a very general concept that can be
# extended in a variety of ways to aid prediction tasks from imaging.
    randspc = rnorm( mydim, mean=2, sd=0.25 )
    if ( myct == 1 & mydim == 3 ) {
      useconstspacing = TRUE
      if ( useconstspacing ) {
        print("WARNING! we are making the 3D problem easier for now by setting constant spacing")
        print("to disable this warning, set useconstspacing to FALSE in this if statement")
      }
    }
    if ( useconstspacing ) randspc = rep( 2 , mydim ) # this makes the problem easier
    img = makeImage( idim, voxval = 0, spacing = randspc )
    msk = img + 1
    spat = imageDomainToSpatialMatrix( msk, msk )
    spatx = makeImage( dim(msk), spat[,1] ) # mean(spatx); randspc
    spaty = makeImage( dim(msk), spat[,2] )
    if ( mydim == 3 ) {
      spatz = makeImage( dim(msk), spat[,3] )
      spatlist = list( spatx, spaty, spatz )
    } else spatlist = list( spatx,spaty )
    pts = matrix( nrow = 1, ncol = mydim )
    ctr = idim / 2
    ptctr = antsTransformIndexToPhysicalPoint( img, ctr )
    pts[1, ] = ptctr
    ptsi = makePointsImage( pts, msk, radius = baserad )
    ptsi = ptsi + makePointsImage( pts, msk, radius = baserad + plusrad[ct] )
    if ( mydim == 2 ) {
      # stretch
      txStretch = createAntsrTransform( "AffineTransform", dim=2 )
      params = getAntsrTransformParameters( txStretch )
      params[1] = rnorm( 1, 1, 0.2 )
      setAntsrTransformParameters(txStretch, params)
      # random rotation
      myradians = rnorm(1,0,180) / 180
      mytrans = rnorm( length(idim), 0, round(baserad)*0.5 )
      txRotate <- createAntsrTransform( type="Euler2DTransform",
        parameters = c(myradians,mytrans), fixed.parameters = ptctr )
      tx = composeAntsrTransforms(list( txRotate, txStretch))
      ptsi = applyAntsrTransformToImage( tx, ptsi, ptsi )
      # now - we can store tx ground truth parameters
      txdf[myct,] = c( params[1], getAntsrTransformParameters( txRotate ),
        getAntsrTransformFixedParameters( txRotate ) )
      }
    if ( mydim == 3 ) {
      # stretch
      txStretch = createAntsrTransform( "AffineTransform", dim=3 )
      params = getAntsrTransformParameters( txStretch )
      params[1] = rnorm( 1, 1, 0.2 ) # **NON-uniform**
      setAntsrTransformParameters(txStretch, params)
      # random rotation
      myradians = rnorm(3,0,180) / 180
      mytrans = rnorm( length(idim), 0, round(baserad)*0.5 )
      txRotate <- createAntsrTransform( type="CenteredEuler3DTransform",
        parameters = c(myradians,ptctr,mytrans)  )
      tx = composeAntsrTransforms(list( txRotate, txStretch))
      ptsi = applyAntsrTransformToImage( tx, ptsi, ptsi )
      # now - we can store tx ground truth parameters
      txdf[myct,] = c( params[1], getAntsrTransformParameters( txRotate )  )
      print( txdf[myct,] )
      }
    ptsi[ msk == 1 ] = ptsi[ msk == 1 ] + rnorm( sum(msk==1),  0, 0.1 )
    if ( mydim == 2 )
      plot( ptsi, doCropping=F, nslices=20, axis=2, window.img=c(0,max(ptsi)) )
    if ( mydim == 3 )
      plot( ptsi, doCropping=F, slices=seq(idim[2]*0.25,idim[3]*0.8,by=2), axis=2, window.img=c(0,max(ptsi)), ncolumns=8 )
    # antsr framework
    ofn = paste( odir, "/singlechannel/", classes[ct], "/", sep='' )
    dir.create( ofn, showWarnings = FALSE, recursive = TRUE, mode = "0777")
    ofn = paste( odir, "/singlechannel/", classes[ct], "/sphere", k, classes[ct], ext, sep='' )
    antsImageWrite( ptsi, ofn )
    ptsm = mergeChannels( lappend( ptsi, spatlist ) )       # multichannel version
    ofn = paste( odir, "/multichannel/", classes[ct], "/", sep='' )
    dir.create( ofn, showWarnings = FALSE, recursive = TRUE, mode = "0777")
    ofn = paste( odir, "/multichannel/", classes[ct], "/sphere", k, classes[ct], ext, sep='' )
    antsImageWrite( ptsm, ofn )
    # numpy framework
    ofn = paste( odir, "/numpysinglechannel/", classes[ct], "/", sep='' )
    dir.create( ofn, showWarnings = FALSE, recursive = TRUE, mode = "0777")
    ofn = paste( odir, "/numpysinglechannel/", classes[ct], "/sphere", k, classes[ct], ext2, sep='' )
    writeANTsImageToNumpy( ptsi, ofn )
    ptsm = mergeChannels( lappend( ptsi, spatlist ) )       # multichannel version
    ofn = paste( odir, "/numpymultichannel/", classes[ct], "/", sep='' )
    dir.create( ofn, showWarnings = FALSE, recursive = TRUE, mode = "0777")
    ofn = paste( odir, "/numpymultichannel/", classes[ct], "/sphere", k, classes[ct], ext2, sep='' )
    writeANTsImageToNumpy( ptsm, ofn )
    comdf[ myct, ] = getCenterOfMass( ptsi )
    myct = myct + 1
    }
  }
ofn = paste( odir, "/spheres", mydim, "CoM.csv", sep='' )
write.csv( comdf, ofn, row.names=F )
ofn = paste( odir, "/spheres", mydim, "Transforms.csv", sep='' )
write.csv( txdf, ofn, row.names=F )
