import numpy as np
import numpy.linalg as LA   # 导入Numpy的线性代数库

x = np.arange(0, 1, 0.1)    # 生成步长为0.1的十个数
print(x)

x1 = LA.norm(x, 1)          # 计算1的范式
x2 = LA.norm(x, 2)          # 计算2的范式
xa = LA.norm(x, np.inf)     # 计算无穷范式

print(x1)
print(x2)
print(xa)