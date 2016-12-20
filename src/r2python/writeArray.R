library( ANTsR )
n = 3
thdimg = makeImage( rep(n,3), rnorm( n^3 ) )
antsImageWrite( thdimg, '/tmp/tenbytenbytenfloat.mhd' )
