args <- commandArgs(trailingOnly = TRUE)
library( ANTsR )
library( jpeg )
library( ANTsRNpy )
# run this in iTensorFlow base directory
temp = unlist(strsplit(getwd(),"/"))
if ( temp[ length( temp ) ] != "iTensorFlow" )
  stop("run this script within the iTensorFlow base dir")
mydim = 3
mydim = round( as.numeric( args[1] ) )
if ( mydim == 3 ) cbp = FALSE else cbp = TRUE
if ( mydim == 2 ) n = c( 100, 25 ) else n = c( 10, 5 )
idim = rep( 32, mydim )
odir = paste( "data/dim", length(idim), "D/segmentation/spheresRad/", c("train/multichannel","test/multichannel"),"/", sep='' )
img = makeImage( idim, voxval = 0, spacing = rep(1, length(idim)) )
baserad = 4
patchRad = 5
msk = img * 0 + 1
spatmask = imageDomainToSpatialMatrix( img, msk )
lo = 10000
print( odir )
doPatches = FALSE
for ( ct in 1:length( odir ) ) {
  myct = 0
  if ( ct == 1 ) {
    nptch = 10
    print("training data")
    myran = TRUE
    } else {
      print("testing data")
      nptch = 10
      myran = FALSE
    }
  for ( k in lo:(lo+n[ct]-1) ) {
    r1 = rnorm( 1, baserad, 1 )
    r2 = rnorm( 1, round( baserad * 0.75 ) , 1 )
    sim = simulateSphereData( img, radius = c( r1, r1+r2 ),
      noiseLevel = c(0, 0.2 ), positionNoiseLevel = c( 0, 2 ), classByPosition = cbp )
    mymask = morphology( getMask( sim$image ), "dilate", 2 )
    if ( doPatches ) {
      patches = multiChannelImageToPatches( sim$mcimage, mask = mymask, radius = patchRad,
          groundTruth = sim$groundTruth$labels[ mymask == 1 ],
          npatches = nptch, randomize = myran )
      if ( ct == 1 & k == lo ) print( patches$patches[[1]] )
      if ( k == lo ) mydf = patches$patchSummary else mydf = rbind( mydf, patches$patchSummary )
      for ( ww in 1:length( patches$patches ) ) {
        ofn = paste( odir[ct], "/", sep='' )
        dir.create( ofn, showWarnings = FALSE, recursive = TRUE, mode = "0777")
        www = stringr::str_pad( ww, 6, pad = "0")
        ofn = paste( odir[ct], "/sphere", k,"_",www,"Image.npy", sep='' )
        temp = capture.output( writeANTsImageToNumpy( patches$patches[[ ww ]], ofn ) )
        myct = myct + 1
        }
      } else {
        ofn = paste( odir[ct], "/", sep='' )
        dir.create( ofn, showWarnings = FALSE, recursive = TRUE, mode = "0777")
        ofn = paste( odir[ct], "/sphereImage", k,".npy", sep='' )
        temp = capture.output( writeANTsImageToNumpy( sim$mcimage, ofn ) )
        ofn = paste( odir[ct], "/sphereSeg", k,".npy", sep='' )
        temp = capture.output( writeANTsImageToNumpy( sim$groundTruthImage, ofn ) )
        myct = myct + 1
      }
    }
    if ( doPatches ) {
      if ( myct != nrow( mydf ) ) stop("Error")
      ofn = paste( odir[ct], "/spheres", mydim, "Segmentation.csv", sep='' )
      write.csv( mydf[,-c(1:2)], ofn, row.names=F )
    }
  }
