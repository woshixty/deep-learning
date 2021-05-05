import numpy as np

C = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
TrC = np.trace(C)

D = C - 2
TrCT = np.trace(C.T)
TrCD = np.trace(np.dot(C, D))
TrDC = np.trace(np.dot(D, C))

print(TrC)
print(TrCT)
print(TrCD)
print(TrDC)