import matplotlib.pylab as pl
import numpy as np

# 固定学习率与自适应学习率对比图
x_relu65 = np.linspace(0, 10000, 1000)
y_relu65 = np.loadtxt('./activation_function/relu85.data')

x_relu67 = np.linspace(0, 10000, 1000)
y_relu67 = np.loadtxt('./activation_function/reluS80.data')

# x_relu61 = np.linspace(0, 10000, 1000)
# y_relu61 = np.loadtxt('./activation_function/relu66.data')

pl.plot(x_relu65, y_relu65, 'b', label='adaptive learning rate')
pl.plot(x_relu67, y_relu67, 'r', label='fixed learning rate')
pl.xlabel('Iteration')
pl.ylabel('loss')
pl.legend()
pl.show()