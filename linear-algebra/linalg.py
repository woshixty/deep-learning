import numpy as np

# np.linalg.eigvals(a)---计算矩阵的特征值
# np.linalg.eig(a)---返回包含特征值和对应特征向量的元组
a = np.array([[1, 2], [3, 4]])  # 示例矩阵
A1 = np.linalg.eigvals(a)       # 得到特征值
A2, V1 = np.linalg.eig(a)       # 其中A2也是特征值，B为特征向量
print(A1)
print(A2)
print(V1)