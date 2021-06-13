import matplotlib.pyplot as plt
import numpy as np
import time


L = np.array([[1.73, 0.00, 0.00], 
[1.73, 1.41, 0.00], 
[2.89, 2.83, 0.82]])
LL = L@L.T
rhs  = np.array([5.000, 9.000, 17.000])

b0 = np.linalg.solve(L, rhs)
print(b0)
b = np.linalg.solve(L.T, b0)
print(b, LL@b)
# b0 [2.89017341 2.83687943 0.75491469]
# b [-0.03149329  0.1641866   0.92062768] [ 5.  9. 17.]
# sol = np.linalg.solve(LL, rhs) #[9.75777426 0.27579541 0.94069045]
# print(sol, LL @ sol)
exit()

x,y, y_, y_sgd = [], [], [], []
a,b,c= 3.5, 4.3, 3
n_sample = 100


# Generate test data
for i in range(n_sample):
    idx = i % 100
    x += [idx]
    y += [a + b*idx + c*idx*idx + 10000* np.random.randint(0,100)/100.0]
x = np.array(x)
y = np.array(y)
# Reference solution using analytic LLS

A = np.vstack([np.ones(len(x)), x, x*x]).T
# t0 = time.time()
for k in range(1000):
    a_, b_, c_ = np.linalg.lstsq(A, y, rcond=None)[0]
# t1 = time.time()
# print(365*365*(t1-t0)/1000)
for i in range(n_sample):
    y_ += [a_ + b_*x[i] + c_*x[i]*x[i]] 
y_ = np.array(y_)

# Our own sgd

params=np.array([0,0,0])
lr = 0.01
for k in range(10):
    for i in range(n_sample):
        y_pred = params[0] + params[1]*x[i] + params[2]*x[i]*x[i]
        err = (y[i] - y_pred)
        J = err*err
        Jgrad = -2*err*np.array([1, x[i], x[i]*x[i]])
        params = params -lr*Jgrad/np.linalg.norm(Jgrad)
# fwd pass:
for i in range(n_sample):
    y_sgd += [params[0] + params[1]*x[i] + params[2]*x[i]*x[i]] 

plt.plot(x,y, "r*")
plt.plot(x,y_, "b-")
plt.plot(x,y_sgd, "b+")
plt.show()
