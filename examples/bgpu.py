import os
os.environ["BGPU"] = "1" if os.getenv("CPU", "0") == "0" else "0"
os.environ["DEBUG"] = "7"

from tinygrad import dtypes
from tinygrad import Tensor

dtype = dtypes.float32

height = 16
width  = height

v0 = []
v1 = []
for i in range(height * width):
  v0.append(i + 1)
  v1.append(i * 2 + 1)

t0 = Tensor(v0, dtype=dtype).reshape(height, width)
t1 = Tensor(v1, dtype=dtype).reshape(width, height)

# t2 = t0.sum(axis=0)
t2 = (t0 @ (t1 + 1))
print(t2.numpy())
