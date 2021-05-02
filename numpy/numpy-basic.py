import numpy as np

# num_list = [3.14, 1, 3, 92]
# nd = np.array(num_list)
# print(num_list)
# print(nd)

# num_list = [[3.14, 1, 3, 92], [1, 2, 3, 4]]
# nd = np.array(num_list)
# print(num_list)
# print(nd)
# nd = np.random.random([3, 3])
# print(nd)

# 3×3矩阵，矩阵元素均为0
# nd1 = np.random.random([3, 3])
# print(nd1)
# numpy.savetxt(X=nd1, fname='test.txt')
# nd0 = np.loadtxt('test.txt')
# print(nd0)

# # 3×3矩阵，矩阵元素均为1
# nd2 = np.ones([3, 3])
# print(nd2)
#
# # 3阶的单位矩阵
# nd3 = np.eye(3)
# print(nd3)
#
# # 3阶对角矩阵
# nd4 = np.diag([1, 2, 3])
# print(nd4)

# # 从0到10，每次加一生成的数组
# print(np.arange(10))
# # 从3到10，每次加一生成的数组
# print(np.arange(3, 10))
# # 从1到11（不等于11），每次加二生成的数组
# print(np.arange(1, 11, 2))
# # 从9一直到-1（不等于-1），每次减一生成的数组
# print(np.arange(9, -1, -1))


# nd = np.diag([1, 2, 3, 4, 5, 6, 7, 8])
# print(nd)
#
# # 获取第四个元素，一个元素即一个数组，起始元素为0
# print("nd[4]:")
# print(nd[4])
#
# # 截取一段数据，从3开始，小于6
# print("nd[3:6]:")
# print(nd[3:6])
#
# # 截取固定间隔数据，从3开始，小于6，每次加二
# print("nd[1:6:2]:")
# print(nd[1:6:2])
#
# # 倒序取数
# print("nd[::-1]:")
# print(nd[::-1])

# nd = np.arange(25).reshape([5, 5])
# print(nd)
#
# # 截取多维数组一个区域内的数据，如下：截取2到3行，2到四列的数据
# print("nd[1:3, 1:2]:")
# print(nd[1:3, 1:4])
#
# # 截取多维数组一个区域内的数据，生成一个一维数组
# print("nd[(nd > 3) & (nd < 10)]:")
# print(nd[(nd > 3) & (nd < 10)])
#
# # 截取多维数组的2、3行
# print("nd[1:3, :]:")
# print(nd[1:3, :])
#
# # 截取多维数组的2、3列
# print("nd[:, 1:3]:")
# print(nd[:, 1:3])


nd = np.arange(9).reshape([3, 3])
print(nd)

# 矩阵砖置
print("矩阵转置：")
print(np.transpose(nd))

# 矩阵的乘法
print("矩阵的乘法：")
a = np.arange(12).reshape([3, 4])  # 三行四列的矩阵
b = np.arange(8).reshape([4, 2])  # 四行两列的矩阵
print(a.dot(b))

# 求矩阵的迹
print("a矩阵的迹：")
print(a.trace())

# 计算行列式
print("计算行列式：")
c = np.arange(4).reshape([2, 2])
print(np.linalg.det(c))

# 计算逆矩阵
print("c的逆矩阵：")
print(np.linalg.solve(c, np.eye(2)))


