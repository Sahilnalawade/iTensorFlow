library( ANTsR )
library( jpeg )
library( ANTsRNpy )
# run this in iTensorFlow base directory
temp = unlist(strsplit(getwd(),"/"))
if ( temp[ length( temp ) ] != "iTensorFlow" )
  stop("run this script within the iTensorFlow base dir")
n = c(1000,200)
mydim = 2
idim = rep( 32, mydim )
odir = paste( "data/dim", length(idim), "D/regression/spheresRad/", c("train/singlechannel","test/singlechannel"),"/", sep='' )
img = makeImage( idim, voxval = 0,
  spacing = rep(1, length(idim)) )
msk = img + 1
spat = imageDomainToSpatialMatrix( msk, msk )
spatx = makeImage( dim(msk), spat[,1] )
spaty = makeImage( dim(msk), spat[,2] )
spatlist = list( spatx,spaty )
baserad = 3
for ( ct in 1:2 ) {
  myct = 1
  comdf = data.frame( x=0, y=0, r1=0, r2=0 )
  for ( k in 1000:(1000+n[ct]-1) ) {
    pts = matrix( nrow = 1, ncol = mydim )
    ctr = idim / 2
    pts[1, ] = antsTransformIndexToPhysicalPoint( img, ctr ) +
      rnorm( length(idim), 0, 3 )
    r1 = rnorm(1,baserad,1)
    r2 = rnorm(1,baserad,1)
    ptsi = makePointsImage( pts, msk, radius = r1 )
    ptsi = ptsi + makePointsImage( pts, msk, radius = r2  )
    comdf[ myct, ] = getCenterOfMass( ptsi )
    ptsi[ msk == 1 ] = ptsi[ msk == 1 ] + rnorm( sum(msk==1),  0, 0.1 )
    # antsr framework
    ofn = paste( odir[ct], "/", sep='' )
    dir.create( ofn, showWarnings = FALSE, recursive = TRUE, mode = "0777")
    ptsj = antsImageClone( iMath( ptsi, "Normalize") * 255, "unsigned char" )
    ofn = paste( odir[ct], "/sphere", k, ".npy", sep='' )
    writeANTsImageToNumpy( ptsi, ofn )
#    antsImageWrite( ptsj, ofn )
#    plot( ptsj, doCropping=F, nslices=20, axis=2, window.img=c(0,max(ptsj)) )
    myct = myct + 1
    }
    ofn = paste( odir[ct], "/spheres", mydim, "Radius.csv", sep='' )
    write.csv( comdf, ofn, row.names=F )
  }
