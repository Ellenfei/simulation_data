import matplotlib.pylab as pl
import numpy as np

#relu与sigmoid函数/隐藏层个数/动态与固定学习率
# print(480/500)
# print(20/500)
# print(460/500)
# print(40/500)
print(476/500)
DR_RULE = 0.964
DR_SIGMOID = 0.92
DR_8 = 0.964
DR_6 = 0.944
DR_10 = 0.952
DR_D = 0.964
DR_S = 0.942
interval = 0.4
bar_width = 0.2
N = 5

x = (0, bar_width/2, interval, interval+bar_width/2, interval+bar_width, interval*2+bar_width/2, interval*2+bar_width)
y = (DR_RULE, DR_SIGMOID, DR_8, DR_6, DR_10,  DR_D, DR_S)


fig, ax = pl.subplots()
index = np.arange(N)
opacity = 1
# 第一个参数左边缘横坐标；第二个参数每根柱子的高度，第三个参数所有柱子的宽度
rects1 = pl.bar(0, DR_RULE, bar_width/2, alpha=opacity, color='b', label='ReLu')
rects2 = pl.bar(bar_width/2, DR_SIGMOID, bar_width/2, alpha=opacity, color='r', label='Sigmoid')
rects3 = pl.bar(interval, DR_8, bar_width/2, alpha=opacity, color='k', label='Number=8')
rects4 = pl.bar(interval+bar_width/2, DR_6, bar_width/2, alpha=opacity, color='y', label='Number=6')
rects5 = pl.bar(interval+bar_width, DR_10, bar_width/2, alpha=opacity, color='C', label='Number=10')
rects6 = pl.bar(interval*2+bar_width/2, DR_D, bar_width/2, alpha=opacity, color='g', label='adaptive learning rate')
rects7 = pl.bar(interval*2+bar_width, DR_S, bar_width/2, alpha=opacity, color='m', label='fixed learning rate')

# pl.xlabel('Group')
pl.ylabel('detection rate')
pl.xticks((bar_width/4, interval+bar_width/2, interval*2+bar_width*0.75, interval*5),
          ('activation function', 'the number of hidden layer', 'learning rate'), fontsize=7)
# pl.xticks(index*interval+bar_width/2, ('activation function', 'the number of hidden layer', 'learning rate'), fontsize=7)
for a, b in zip(x, y):
    pl.text(a, b+0.006, '%.3f' % b, ha='center', va='bottom', fontsize=7)

pl.ylim(0, 1)
pl.legend(loc='upper right')

pl.tight_layout()
pl.show()

