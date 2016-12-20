import numpy as np
n = 3
filename = '/tmp/tenbytenbytenfloat.raw'
with open(filename, 'rb') as f:
  data = np.fromfile(f, dtype=np.float32)
  array = np.reshape(data, [n, n, n])

# this array is the "by slice" transpose of the R result ...
