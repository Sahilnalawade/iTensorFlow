library( ANTsR )
library( jpeg )
# run this in iTensorFlow base directory
temp = unlist(strsplit(getwd(),"/"))
if ( temp[ length( temp ) ] != "iTensorFlow" )
  stop("run this script within the iTensorFlow base dir")
n = 50
if ( ! exists("mydim") ) mydim = 2
if ( mydim == 2 ) ext = ".jpg" else ext = ".nii.gz"
if ( mydim == 2 ) ext = ".nii.gz" else ext = ".nii.gz"
idim = rep( 64, mydim )
odir = paste( "data/dim", length(idim), "D/classification/spheres", sep='' )
img = makeImage( idim, voxval = 0,
  spacing = rep(1, length(idim)) )
msk = img + 1
spat = imageDomainToSpatialMatrix( msk, msk )
spatx = makeImage( dim(msk), spat[,1] )
spaty = makeImage( dim(msk), spat[,2] )
if ( mydim == 3 ) {
  spatz = makeImage( dim(msk), spat[,3] )
  spatlist = list( spatx, spaty, spatz )
} else spatlist = list( spatx,spaty )
baserad = 5
plusrad = c( 1, 2 )
classes = c("class1","class2")
if ( mydim == 3 ) comdf = data.frame( x=0, y=0, z=0 )
if ( mydim == 2 ) comdf = data.frame( x=0, y=0 )
myct = 1
for ( ct in 1:2 ) {
  for ( k in 1000:(1000+n) ) {
    pts = matrix( nrow = 1, ncol = mydim )
    ctr = idim / 2
    pts[1, ] = antsTransformIndexToPhysicalPoint( img, ctr ) +
      round( rnorm( length(idim), 5, 2 ) )
    ptsi = makePointsImage( pts, msk, radius = baserad )
    ptsi = ptsi + makePointsImage( pts, msk, radius = baserad + plusrad[ct] )
    ptsi[ msk == 1 ] = ptsi[ msk == 1 ] + rnorm( sum(msk==1),  0, 0.1 )
    ofn = paste( odir, "/singlechannel/", classes[ct], "/", sep='' )
    dir.create( ofn, showWarnings = FALSE, recursive = TRUE, mode = "0777")
    ofn = paste( odir, "/singlechannel/", classes[ct], "/sphere", k, classes[ct], ext, sep='' )
    antsImageWrite( ptsi, ofn )
    ptsm = mergeChannels( lappend( ptsi, spatlist ) )       # multichannel version
    ofn = paste( odir, "/multichannel/", classes[ct], "/", sep='' )
    dir.create( ofn, showWarnings = FALSE, recursive = TRUE, mode = "0777")
    ofn = paste( odir, "/multichannel/", classes[ct], "/sphere", k, classes[ct], ext, sep='' )
    antsImageWrite( ptsm, ofn )
    comdf[ myct, ] = getCenterOfMass( ptsi )
    myct = myct + 1
#    plot( ptsi, doCropping=F, nslices=20, axis=2, window.img=c(0,max(ptsi)) )
    }
  }
ofn = paste( odir, "/spheres", mydim, "CoM.csv", sep='' )
write.csv( comdf, ofn, row.names=F )
