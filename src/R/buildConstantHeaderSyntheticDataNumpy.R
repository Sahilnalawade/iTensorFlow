library( ANTsR )
library( jpeg )
library( ANTsRNpy )
# run this in iTensorFlow base directory
temp = unlist(strsplit(getwd(),"/"))
if ( temp[ length( temp ) ] != "iTensorFlow" )
  stop("run this script within the iTensorFlow base dir")
n = c( 100, 20 )
mydim = 2
idim = rep( 32, mydim )
odir = paste( "data/dim", length(idim), "D/regression/spheresRad/", c("train/singlechannel","test/singlechannel"),"/", sep='' )
img = makeImage( idim, voxval = 0, spacing = rep(1, length(idim)) )
baserad = 3
for ( ct in 1:2 ) {
  myct = 1
  comdf = data.frame( x=0, y=0, r1=0, r2=0 )
  for ( k in 1000:(1000+n[ct]-1) ) {
    r1 = rnorm( 1, baserad, 1 )
    r2 = rnorm( 1, baserad, 1 )
    sim = simulateSphereData( img, radius = c( r1, r1+r2 ), positionNoiseLevel = c( 0, 2 ) )
    comdf[ myct,  ] = c( sim$centerOfMass, r1, r2 )
    # antsr framework
    ofn = paste( odir[ct], "/", sep='' )
    dir.create( ofn, showWarnings = FALSE, recursive = TRUE, mode = "0777")
    ptsj = antsImageClone( iMath( sim$image, "Normalize") * 255, "unsigned char" )
    ofn = paste( odir[ct], "/sphere", k, "Image.npy", sep='' )
    writeANTsImageToNumpy( sim$image, ofn )
#    antsImageWrite( ptsj, ofn )
#    plot( ptsj, doCropping=F, nslices=20, axis=2, window.img=c(0,max(ptsj)) )
    myct = myct + 1
    }
    ofn = paste( odir[ct], "/spheres", mydim, "Radius.csv", sep='' )
    write.csv( comdf, ofn, row.names=F )
  }
