library( ANTsR )
idim = 3
idim = rep( 32, idim )
img = makeImage( idim, voxval = 0,
  spacing = rep(1, length(idim)) )
msk = img + 1
pts = matrix( nrow = 1, ncol = length( idim ) )
ctr = idim / 2
pts[1, ] = antsTransformIndexToPhysicalPoint( img, ctr )
ptsi = makePointsImage( pts, msk, radius = 5 )
ptsi = ptsi + makePointsImage( pts, msk, radius = 6 )
ptsi[ msk == 1 ] = ptsi[ msk == 1 ] + rnorm( sum(msk==1),  0, 0.1 )
plot( ptsi, doCropping=F, nslices=20, axis=2, window.img=c(0,max(ptsi)) )
