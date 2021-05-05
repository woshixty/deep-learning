import numpy as np

a = np.array([1, 2, 3, 8])
print(a.size)
print(a[0], a[-1])

A = np.array([[1, 2, 3], [4, 5, 6]])
print(A)
print(A.size)       # 显示矩阵元素总个数
print(A.shape)      # 显示矩阵的行数和列数
print(A[0, 0], A[0, 1])
print(A[1, :])      # 打印矩阵第二行


# np.arange(num)---生成一个从0到num-1步数为1的一维数组
# ndarray.reshape((N,M,...))---将ndarray转化为N*M*...的多维数组
B = np.arange(16).reshape((2, 2, 4))    # 生成一个2*2*4的三阶矩阵
print(B)
print(B.size)       # 显示B的元素总数
print(B.shape)      # 显示B的维度
print(B[0, 0, 0])

C = np.array([[1, 2, 3], [4, 5, 6]])
CT = C.T
print(C)
print(CT)

