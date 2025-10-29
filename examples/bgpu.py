import os
# os.environ["BGPU"] = "1"
os.environ["DEBUG"] = "7"
os.environ["NOOPT"] = "0"

from tinygrad import dtypes
from tinygrad import Tensor

dtype = dtypes.int32

height = 2
width  = height

v0 = []
v1 = []
for i in range(height * width):
  v0.append(i)
  v1.append(i * 2)

t0 = Tensor(v0, dtype=dtype).reshape(height, width)
t1 = Tensor(v1, dtype=dtype).reshape(width, height)

# t2 = t0.sum(axis=0)
t2 = (t0 + t1 * 2)
print(t2.numpy())
