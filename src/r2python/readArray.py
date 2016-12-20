import numpy as np
n = 3
filename = '/tmp/tenbytenbytenfloat.raw'
with open(filename, 'rb') as f:
  data = np.fromfile(f, dtype=np.float32)
  array = np.reshape(data, [n, n, n])

# this array is the "by slice" transpose of the R result ...

# now try to read a RcppCNPy object
m = np.load("/tmp/randmat.npy")
m = np.reshape( np.load("/tmp/randmat3.npy"), [n, n, n])
