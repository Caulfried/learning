# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 17:14:15 2019

@author: Caulfried
"""

import numpy as np
import pandas as pd
###模型
def function(x1,x2):
    R1 = 0.25
    S1 = 0.25
    R2 = 0.75
    S2 = 0.75
    A = 0.025
    S = 0.5
    y = 1/3*x1**3-(R1+S1)*1/2*x1**2+(R1*S1)*x1+1/3*x2**3-(R2+S2)*1/2*x2**2+(R2*S2)*x2+A*np.sin(2/S*np.pi*x1*x2)
    return(y)
###画出模型的3D图
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
fig = plt.figure()
ax = Axes3D(fig)
X1 = np.arange(0,1,0.1)
X2 = np.arange(0,1,0.1)
x1, x2 = np.meshgrid(X1,X2)
R1 = 0.25
S1 = 0.25
R2 = 0.75
S2 = 0.75
A = 0.025
S = 0.5
y = 1/3*x1**3-(R1+S1)*1/2*x1**2+(R1*S1)*x1+1/3*x2**3-(R2+S2)*1/2*x2**2+(R2*S2)*x2+A*np.sin(2/S*np.pi*x1*x2)

plt.xlabel('x1')
plt.ylabel('x2')
ax.plot_surface(x1, x2, y, rstride=1, cstride=1, cmap='rainbow')
plt.show()

###使用试验次数与水平数均为30次的的两因素均匀设计
design = pd.read_csv('C:/Users/Caulfried/Desktop/均匀设计.csv')
design_trans = (2*design-1)/(2*30)
y1 = function(design_trans['x1'],design_trans['x2'])
###随机产生一个两因素设计
#import random
#x1 = list(np.arange(1,31))
#x2 = list(np.arange(1,31))
#random.shuffle(x1)
#random.shuffle(x2)
#design_ran = pd.DataFrame({'x1':x1,'x2':x2})
#design_ran.to_csv('C:/Users/Caulfried/Desktop/design_ran.csv', index=False)
design_ran = pd.read_csv('C:/Users/Caulfried/Desktop/design_ran.csv')
design_ran_trans = (2*design_ran-1)/(2*30)###对各因素水平进行标准化处理
y0 = function(design_ran_trans['x1'],design_ran_trans['x2'])

###使用神经网络进行训练
import tensorflow as tf

n_inputs = 2
n_hidden1 = 15
n_hidden2 = 10
n_hidden3 = 10
n_hidden4 = 10
n_hidden5 = 10
n_output = 1

input_x = np.array(design_trans)
input_y = np.array(y1)
dataset = tf.data.Dataset.from_tensor_slices((input_x,input_y))
dataset = dataset.shuffle(30).batch(10).repeat()
iterator = dataset.make_one_shot_iterator()
batch_one_element = iterator.get_next() 
###创建占位符
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(tf.int64, shape=(None), name='y')
###层的搭建
from tensorflow.contrib.layers import fully_connected
with tf.name_scope('ann'):
    hidden1 = fully_connected(X, n_hidden1, scope='hidden1')
    hidden2 = fully_connected(hidden1, n_hidden2, scope='hidden2')    
    y_ = fully_connected(hidden2, n_output, scope='output', activation_fn=None)
###损失函数
with tf.name_scope("loss"):
    mse = tf.reduce_mean(tf.square(y_-input_y))
###优化器
learning_rate = 0.01
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(mse)

###初始化器
init = tf.global_variables_initializer()
saver = tf.train.Saver()    

n_epochs = 100
batch_size = 10

###模型训练与训练集上的结果
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(3):
            X_batch, y_batch = sess.run(batch_one_element)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        accuracy_epoch = tf.reduce_mean(tf.square(y_-input_y))
        acc_epoch = accuracy_epoch.eval(feed_dict={X: input_x, y: input_y})
        print('epoch_mse:',acc_epoch)    
    accuracy_train = tf.reduce_mean(tf.square(y_-input_y))
    acc_train = accuracy_train.eval(feed_dict={X: input_x, y: input_y})
    print('Train_mse:',acc_train)
    save_path = saver.save(sess, "D:/学习/研一上/tensorflow/my_model_final.ckpt")

###测试集上的结果
with tf.Session() as sess:
    saver.restore(sess, "D:/学习/研一上/tensorflow/my_model_final.ckpt")
    X_test = np.array(Xtest)
    y_test = np.array(ytest)
    y_pred = y_.eval(feed_dict={X: X_test})
    accuracy = tf.reduce_mean(tf.square(y_pred-y_test))
    acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
    print('Test_mse:',acc_test)

pd.DataFrame(y_pred,y_test)
lst = [0.002115327519995311，0.0016787971791436824]
0.0027628,0.0035
















