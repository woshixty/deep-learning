import theano
import theano.tensor as T
from theano import shared
import numpy as np

# 定义一个共享变量，并初始化为0
state = shared(0)
inc = T.iscalar('inc')
accumulator = theano.function([inc], state, updates=[(state, state+inc)])
# 打印state的初始值
print(state.get_value())
accumulator(1)      # 进行了一次函数调用
# 函数返回以后，state的值发生了变化
print(state.get_value())