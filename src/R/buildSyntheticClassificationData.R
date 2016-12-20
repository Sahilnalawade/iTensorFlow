library( ANTsR )
# run this in iTensorFlow base directory
temp = unlist(strsplit(getwd(),"/"))
if ( temp[ length( temp ) ] != "iTensorFlow" )
  stop("run this script within the iTensorFlow base dir")
n = 50
idim = 3
idim = rep( 64, idim )
odir = paste( "data/dim", length(idim), "D/classification/spheres", sep='' )
img = makeImage( idim, voxval = 0,
  spacing = rep(1, length(idim)) )
msk = img + 1
baserad = 5
plusrad = c( 1, 2 )
classes = c("class1","class2")
if ( length( idim ) == 3 ) comdf = data.frame( x=0, y=0, z=0 )
if ( length( idim ) == 2 ) comdf = data.frame( x=0, y=0 )
myct = 1
for ( ct in 1:2 ) {
  for ( k in 1000:(1000+n) ) {
    pts = matrix( nrow = 1, ncol = length( idim ) )
    ctr = idim / 2
    pts[1, ] = antsTransformIndexToPhysicalPoint( img, ctr ) +
      round( rnorm( length(idim), 5, 2 ) )
    ptsi = makePointsImage( pts, msk, radius = baserad )
    ptsi = ptsi + makePointsImage( pts, msk, radius = baserad + plusrad[ct] )
    ptsi[ msk == 1 ] = ptsi[ msk == 1 ] + rnorm( sum(msk==1),  0, 0.1 )
    ofn = paste( odir, "/", classes[ct], "/", sep='' )
    dir.create( ofn, showWarnings = FALSE, recursive = TRUE, mode = "0777")
    ofn = paste( odir, "/", classes[ct], "/sphere", k, classes[ct], ".nii.gz", sep='' )
    antsImageWrite( ptsi, ofn )
    comdf[ myct, ] = getCenterOfMass( ptsi )
    myct = myct + 1
#    plot( ptsi, doCropping=F, nslices=20, axis=2, window.img=c(0,max(ptsi)) )
    }
  }
ofn = paste( odir, "/spheres", length( idim ), "CoM.csv", sep='' )
write.csv( comdf, ofn, row.names=F )
