from scipy.io import savemat
import numpy as np
a = np.arange(20).reshape(-1,1)
b = np.arange(20).reshape(-1,1)
print(a)
mdic = {"a": a, "b": b}
savemat("matlab_matrix.mat", mdic)