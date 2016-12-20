library( ANTsR )
n = 3
thdimg = makeImage( rep(n,3), rnorm( n^3 ) )
antsImageWrite( thdimg, '/tmp/tenbytenbytenfloat.mhd' )


library( RcppCNPy )
thdarr = as.array( thdimg )
npySave( "/tmp/randmat.npy", thdarr[,,2] )
# this will fail to write an array but will write a numeric vector that can be reshaped
npySave( "/tmp/randmat3.npy", thdarr )
