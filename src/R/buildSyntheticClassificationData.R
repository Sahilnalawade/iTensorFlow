library( ANTsR )
# run this in iTensorFlow base directory
odir = "data/dim3D/classification/spheres"
n = 50
idim = 3
idim = rep( 64, idim )
img = makeImage( idim, voxval = 0,
  spacing = rep(1, length(idim)) )
msk = img + 1
baserad = 5
plusrad = c( 1, 2 )
classes = c("class1","class2")
for ( ct in 1:2 ) {
  for ( k in 10:(10+n) ) {
    pts = matrix( nrow = 1, ncol = length( idim ) )
    ctr = idim / 2
    pts[1, ] = antsTransformIndexToPhysicalPoint( img, ctr ) +
      round( rnorm( length(idim), 5, 2 ) )
    ptsi = makePointsImage( pts, msk, radius = baserad )
    ptsi = ptsi + makePointsImage( pts, msk, radius = baserad + plusrad[ct] )
    ptsi[ msk == 1 ] = ptsi[ msk == 1 ] + rnorm( sum(msk==1),  0, 0.1 )
    ofn = paste( odir, "/", classes[ct], "/sphere", k, classes[ct], ".nii.gz", sep='' )
    antsImageWrite( ptsi, ofn )
#    plot( ptsi, doCropping=F, nslices=20, axis=2, window.img=c(0,max(ptsi)) )
#    Sys.sleep( 1 )
    }
  }
