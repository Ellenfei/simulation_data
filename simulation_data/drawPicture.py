import matplotlib.pylab as pl
import numpy as np

# # 隐藏层函数比较（ReLu和Sigmoid函数）
x1 = np.linspace(0, 10000, 1000)
y1 = np.loadtxt('./activation_function/reluDP66.data')

x2 = np.linspace(0, 10000, 1000)
y2 = np.loadtxt('./activation_function/sigmoid63.data')


pl.plot(x1, y1, 'b', label='ReLu activation function')
pl.plot(x2, y2, 'r', label='Sigmoid activation function')
pl.xlabel('Iteration')
# pl.xlabel('*1000', horizontalalignment=right)
pl.ylabel('loss')
# pl.legend()
# # print(p1)
# pl.show()
#
# # 学习率动态和固定值比较
# x3 = np.linspace(0, 10000, 1000)
# y3 = np.loadtxt('learning_rate0.data')
#
# x4 = np.linspace(0, 10000, 1000)
# y4 = np.loadtxt('learning_rate_s.data')
#
# x5 = np.linspace(0, 10000, 1000)
# y5 = np.loadtxt('learning_rate_d.data')
#
# x6 = np.linspace(0, 10000, 1000)
# y6 = np.loadtxt('learning_rate_d1.data')
#
# # pl.plot(x3, y3, 'b', label='fixed learning rate')
# pl.plot(x4, y4, 'r', label='fixed learning rate')
# # pl.plot(x5, y5, 'y', label='dynamic learning rate')
# pl.plot(x6, y6, 'b', label='dynamic learning rate')
# pl.xlabel('Iteration')
# pl.ylabel('loss')
# pl.legend()
# pl.show()

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

# 隐藏层神经元数目比较（6/8/10）
x3 = np.linspace(0, 10000, 1000)
y3 = np.loadtxt('hidden6.data')

x4 = np.linspace(0, 10000, 1000)
y4 = np.loadtxt('hidden10_1.data')

x5 = np.linspace(0, 10000, 1000)
y5 = np.loadtxt('hidden8_2.data')

x6 = np.linspace(0, 10000, 1000)
y6 = np.loadtxt('hidden6_3.data')

# pl.plot(x3, y3, 'b', label='hidden6_0')
# pl.plot(x4, y4, 'k', label='hidden10_2')
# pl.plot(x5, y5, 'b', label='hidden8_2')
# pl.plot(x6, y6, 'r', label='hidden6_2')
# pl.xlabel('Iteration')
# pl.ylabel('loss')
# pl.legend()
# pl.show()

x_relu65 = np.linspace(0, 10000, 1000)
y_relu65 = np.loadtxt('./activation_function/relu65.data')

x_relu67 = np.linspace(0, 10000, 1000)
y_relu67 = np.loadtxt('./activation_function/relu67.data')

x_relu61 = np.linspace(0, 10000, 1000)
y_relu61 = np.loadtxt('./activation_function/relu66.data')

x_relu80 = np.linspace(0, 10000, 1000)
y_relu80 = np.loadtxt('./activation_function/relu85.data')

x_relu10 = np.linspace(0, 10000, 1000)
y_relu10 = np.loadtxt('./activation_function/relu106.data')

x_sigmoid60 = np.linspace(0, 10000, 1000)
y_sigmoid60 = np.loadtxt('./activation_function/sigmoid60.data')

x_sigmoid61 = np.linspace(0, 10000, 1000)
y_sigmoid61 = np.loadtxt('./activation_function/sigmoid61.data')

# pl.plot(x_relu64, y_relu64, 'r', label='relu64')
# pl.plot(x_relu65, y_relu65, 'k', label='relu65')
# pl.plot(x_relu67, y_relu67, 'b', label='relu67')
pl.plot(x_relu61, y_relu61, 'b', label='Number=6')
pl.plot(x_relu80, y_relu80, 'r', label='Number=8')
pl.plot(x_relu10, y_relu10, 'k', label='Number=10')
# pl.plot(x_sigmoid61, y_sigmoid61, 'k', label='sigmoid61')
pl.xlabel('Iteration')
pl.ylabel('loss')
pl.legend()
pl.show()