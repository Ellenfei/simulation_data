import sys
import numpy as np
import tensorflow as tf
import thrift
import math
sys.path.append('gen-py')

from generateData import Utils
from data_format import DataService
from data_format.ttypes import *

from thrift import Thrift
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer

class DataServiceHandler:
    def __init__(self):
        self.weight1 = []
        self.biase1 = []
        self.weight2 = []
        self.biase2 = []
    #     n_input = 5
    #     n_hidden = 6
    #     n_output = 1
    #     self.parameters = []
    #     self.parametersData = []
    #     self.parametersData.append([tf.Variable(tf.random_normal([n_input, n_hidden])),
    #                                 tf.Variable(tf.zeros([1, n_hidden]) + 0.1),
    #                                 tf.Variable(tf.random_normal([n_hidden, n_output])),
    #                                 tf.Variable(tf.zeros([1, n_output]) + 0.1)])

    # def ping(self):
    #     print('ping()')

    def buildModel(self, type1, datas):
        if type1 == BussinessType.VIDEO:
            print('视频业务参数')
        elif type1 == BussinessType.ECOMMERCE:
            print('电子商务业务参数')
        elif type1 == BussinessType.EMAIL:
            print('电子邮件业务参数')
        elif type1 == BussinessType.FILE_TRANSFER:
            print('文件传输业务参数')

        # 1.将数据集转换为numpy矩阵
        # 如果一条数据，直接返回参数
        listData = []
        if len(datas) == 1:
            return Parameters(w1=self.weight1, b1=self.biase1, w2=self.weight2, b2=self.biase2)
        for i in range(len(datas)):
            listData.append([datas[i].nodeS, datas[i].nodeD, datas[i].bandwidth, datas[i].delay,
                             datas[i].loss, datas[i].flag])


        toMatrix = np.mat(listData)
        # print(toMatrix)
        x_data = toMatrix[:, 0:5]
        y_data = toMatrix[:, 5]

        # 2.定义网络参数(输入层/隐藏层/输出层神经元个数)
        n_input = 5
        n_hidden = 6
        n_output = 1
        learning_rate = 0.1
        iteration_num = 100
        # 正则化系数
        a = 0.01

        # 3.定义节点准备接收数据
        xs = tf.placeholder(tf.float64, [None, n_input])
        ys = tf.placeholder(tf.float64, [None, n_output])

        # 4.定义神经层（输入层/隐藏层/输出层）
        layer1 = self.add_layer1(xs, n_input, n_hidden, activation_function=tf.nn.relu)
        layer2 = self.add_layer2(layer1, n_hidden, n_output, activation_function=tf.nn.sigmoid)

        # 5.损失函数
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys -layer2), reduction_indices=[1]))

        # 6.损失函数最小化，学习率为0.01
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

        # 7.初始化变量
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        # 8.设置迭代次数，训练模型
        for i in range(iteration_num):
            sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
            # 修改learning rate
            learning_rate = math.exp(-0.001) * learning_rate
            if i == iteration_num/50:
                learning_rate = 0.08
            # print(learning_rate)
            # if (sess.run(loss, feed_dict={xs: x_data, ys: y_data}) < 0.001):
            #     print(i + 1)
            #     break

            if i % 10 == 0:
                print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
        w = np.array([[2.333333333333, 4.66664444444444444444, 5.898866666666], [2.222, 2.333,2.444]])
        b = np.array([1.1111, 1.2222])
        w0 = np.array([[1.555,3.666], [3.444,6.688]])
        b0 = np.array([6.666])
        print(w.dtype)
        ww = w.tolist()
        print(type(ww))
        print(type(ww[0]))
        print(type(ww[0][0]))
        print('*************')
        print((sess.run(self.weights_one)).dtype)
        print(type((sess.run(self.weights_one)).tolist()))
        print(type((sess.run(self.weights_one)).tolist()[0]))
        print(type((sess.run(self.weights_one)).tolist()[0][0]))
        # print(sess.run(self.weights_one).item())
        #
        # return Parameters(w1=self.convertType(sess.run(self.weights_one)),
        #                   b1=self.convertType(sess.run(self.biases_one)),
        #                   w2=self.convertType(sess.run(self.weights_two)),
        #                   b2=self.convertType(sess.run(self.biases_two)))

        # return Parameters(w1=(sess.run(self.weights_one)).tolist(), b1=(sess.run(self.biases_one)).tolist(),
        #                  w2=(sess.run(self.weights_two)).tolist(), b2=(sess.run(self.biases_two)).tolist())
        self.weight1 = (sess.run(self.weights_one)).tolist()
        self.biase1 = (sess.run(self.biases_one)).tolist()
        self.weight2 = (sess.run(self.weights_two)).tolist()
        self.biase2 = (sess.run(self.biases_two)).tolist()
        print((sess.run(self.weights_one)).tolist())
        print(w.tolist())
        return Parameters(w1=w.tolist(), b1=b.tolist(), w2=w0.tolist(), b2=b0.tolist())
        # return Parameters(w1=[[2.3333, 4.6666, 5.8988], [2.222, 2.333,2.444]], b1=[1.1111, 1.2222],
        #                   w2=[[1.555,3.666], [3.444,6.688]], b2=[6.666])

    def convertType(self, arr):

        b = [[] for i in range(arr.shape[0])]
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                b[i].append(arr[i][j].item())
                # print(type(b[i][j]))
        return b


    # BP神经层函数
    def add_layer1(self, input, in_size, out_size, activation_function=None):
        # 权重和偏置量
        self.weights_one = tf.Variable(tf.random_normal([in_size, out_size], dtype=tf.float64),dtype=tf.float64)
        self.biases_one = tf.Variable(tf.zeros([1, out_size], dtype=tf.float64) + 0.1, dtype=tf.float64)

        # 输出
        out = tf.matmul(input, self.weights_one) + self.biases_one
        # 激活函数
        if activation_function is None:
            outputs = out
        else:
            outputs = activation_function(out)
        return outputs
    def add_layer2(self, input, in_size, out_size, activation_function=None):
        # 权重和偏置量
        self.weights_two = tf.Variable(tf.random_normal([in_size, out_size], dtype=tf.float64), dtype=tf.float64)
        self.biases_two = tf.Variable(tf.zeros([1, out_size], dtype=tf.float64) + 0.1, dtype=tf.float64)
        # 输出
        out = tf.matmul(input, self.weights_two) + self.biases_two
        # 激活函数
        if activation_function is None:
            outputs = out
        else:
            outputs = activation_function(out)
        return outputs

bussinessType = 1

datas = []
test1 = Utils()

for i in range(10000):
    datas.append(test1.generateData(True))
    datas.append(test1.generateData(False))
# 定义list接收数据,将数据转换为list

test = DataServiceHandler()
parameters = test.buildModel(bussinessType, datas)

# handler = DataServiceHandler()
# processor = DataService.Processor(handler)
# transport = TSocket.TServerSocket(port=9090)
# tfactory = TTransport.TBufferedTransportFactory()
# pfactory = TBinaryProtocol.TBinaryProtocolFactory()
#
# server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)
#
# print('Starting the server.......')
# server.serve()
# print('done')


