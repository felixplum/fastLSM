import matplotlib.pyplot as plt
import numpy as np
x = []
y = []
y_ = []
a= 3.5
b= 4.3
c= 3
n_sample = 50
for i in range(n_sample):
    idx = i % 100
    x += [idx]
    y += [a + b*idx + c*idx*idx + 10000* np.random.randint(0,100)/100.0]

x = np.array(x)
y = np.array(y)

A = np.vstack([np.ones(len(x)), x, x*x]).T
a_, b_, c_ = np.linalg.lstsq(A, y, rcond=None)[0]

for i in range(n_sample):
    y_ += [a_ + b_*x[i] + c_*x[i]*x[i]] 

y_ = np.array(y_)

plt.plot(x,y, "r*")
plt.plot(x,y_, "b+")
plt.show()
