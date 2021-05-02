import theano
import numpy as np
import theano.tensor as T

# x = T.dmatrix('x')
# y = T.dmatrix('y')
# z = x + y

# 1、下面是多个自变量与多个因变量的函数定义
# x1, y1 = theano.tensor.fscalars('x', 'y')
# z1 = x1 + y1
# z2 = x1*y1
# # 定义x、y为自变量，z1、z2为函数值，因变量
# f = theano.function([x1, y1], [z1, z2])
# # 返回当x=2、y=3的时候，f的因变量z1、z2的值
# print(f(2, 3))

# 2、自动求导
# x = theano.tensor.fscalars('x')     # 定义了一个float类型变量
# y = 1 / (1 + theano.tensor.exp(-x))     # 定义变量y
# dx = theano.grad(y, x)      # 定义偏导数
# f = theano.function([x], dx)        # 定义函数f，输入为x输出为s的偏导数
# print(f(3))     # 计算当x=3时，函数y的偏导数

# 3、更新共享变量参数
# w = theano.shared(1)    # 这里定义了一个共享变量w，其初始值为1
# x = theano.tensor.iscalar('x')
# f = theano.function([x], w, updates=[[w, w+x]])     # 定义函数自变量为x，因变量为w，当函数执行完毕，更新参数w=w+x
# print(f(3))     # 函数输出为w
# print(w.get_value())

# 下面通过一个罗辑回归的完整实例来说明Theano函数的使用方法
rng = np.random

# 为了测试，生成十个样本，没个样本是3维向量，用于训练
N = 10
feats = 3

# randn函数返回一个或一组样本，具有标准正态分布。randn(d0,d1,…,dn)---dn表格每个维度 randint返回随机整数，范围区间为[low,high），包含low，不包含high。randint(low,
# high=None, size=None, dtype=’l’)---low为最小值，high为最大值，size为数组维度大小，dtype为数据类型，默认的数据类型是np.int
D = (rng.randn(N, feats).astype(np.float32), rng.randint(size=N, low=0, high=2).astype(np.float32))

# 声明自变量x、以及每个样本对应的标签y（训练标签）
x = T.matrix("x")
y = T.vector("y")

# 随机初始化参数w、b=0，为共享变量
w = theano.shared(rng.randn(feats), name="w")
b = theano.shared(0., name="b")

# 构造代价函数
p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))  # s激活函数
xent = -y * T.log(p_1) - (1 - y) * T.log(1 - p_1)  # 交叉殇代价函数
cost = xent.mean() + 0.01 * (w ** 2).sum()  # 防止过拟合，权重衰减系数为0.01
gw, gb = T.grad(cost, [w, b])  # 对总代价函数求偏导数

prediction = p_1 > 0.5  # 大于0.5预测值为1
# 训练所需函数
train = theano.function(inputs=[x, y], outputs=[prediction, xent], updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb)))
# 测试阶段函数
predict = theano.function(inputs=[x], outputs=prediction)

# 训练
training_steps = 1000
for i in range(training_steps):
    pred, err = train(D[0], D[1])
    print(err.mean())  # 查看代价函数下降变化过程
