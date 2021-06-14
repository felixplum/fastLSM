import matplotlib.pyplot as plt
import numpy as np
import time
import matplotlib.cm as cm

with open("mat_out.txt", "r") as f:
    lines = f.readlines()

mat = np.zeros((len(lines)))
idx = 0
for l in lines:
    val = l.strip()
    mat[idx] = val
    idx += 1

mat = np.reshape(mat, (365, 365)).T
mat = mat[130:200, 250:]
fig = plt.figure()
ax1 = fig.add_subplot(111)
# Bilinear interpolation - this will look blurry
ax1.imshow(mat, interpolation='nearest')
ax1.set_xlabel("Days [t]")
ax1.set_ylabel("Volume state")

plt.show()