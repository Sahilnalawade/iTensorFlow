library( ANTsR )
library( jpeg )
library( ANTsRNpy )
# run this in iTensorFlow base directory
temp = unlist(strsplit(getwd(),"/"))
if ( temp[ length( temp ) ] != "iTensorFlow" )
  stop("run this script within the iTensorFlow base dir")
n = c(1000,200)
n = c(50,5)
mydim = 2
idim = rep( 32, mydim )
odir = paste( "data/dim", length(idim), "D/segmentation/spheresRad/", c("train/singlechannel","test/singlechannel"),"/", sep='' )
img = makeImage( idim, voxval = 0, spacing = rep(1, length(idim)) )
baserad = 4
patchRad = 5
msk = img * 0 + 1
spatmask = imageDomainToSpatialMatrix( img, msk )
nptch = 10
lo = 10000
print( odir )
for ( ct in 1:length( odir ) ) {
  myct = 0
  if ( ct == 1 ) print("training data") else print("testing data")
  for ( k in lo:(lo+n[ct]-1) ) {
    r1 = rnorm( 1, baserad, 1 )
    r2 = rnorm( 1, round( baserad * 0.75 ) , 1 )
    sim = simulateSphereData( img, radius = c( r1, r1+r2 ),
      noiseLevel = c(0, 0.01 ), positionNoiseLevel = c( 0, 2 ) )
    mymask = morphology( getMask( sim$image ), "dilate", 2 )
    # plot( sim$image, mymask, alpha=0.25, window.img=c(0,2) )
    patches = imageToPatches( sim$image, mask = mymask, radius = patchRad,
      groundTruth = sim$groundTruth$labels[ mymask == 1 ],
      npatches = nptch, randomize = FALSE )
    if ( ct == 1 & k == lo ) print( patches$patches[[1]] )
    patches$patchSummary
    if ( k == lo ) mydf = patches$patchSummary else mydf = rbind( mydf, patches$patchSummary )
    for ( ww in 1:length( patches$patches ) ) {
  #    plot( patches$patches[[ww]], window.img = c(0,2.5), doCropping = F )
  #    patches$patchSummary[ww,]
      ofn = paste( odir[ct], "/", sep='' )
      dir.create( ofn, showWarnings = FALSE, recursive = TRUE, mode = "0777")
      www = stringr::str_pad( ww, 6, pad = "0")
      ofn = paste( odir[ct], "/sphere", k,"_",www,".npy", sep='' )
      temp = capture.output( writeANTsImageToNumpy( patches$patches[[ ww ]], ofn ) )
      myct = myct + 1
      }
    }
    if ( myct != nrow( mydf ) ) stop("Error")
    ofn = paste( odir[ct], "/spheres", mydim, "Segmentation.csv", sep='' )
    write.csv( mydf[,-c(1:(ncol(mydf)-1))], ofn, row.names=F )
  }
