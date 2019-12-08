# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 18:01:39 2019

@author: Caulfried
"""

import numpy as np
import panas as pd

design_ran = pd.read_csv('C:/Users/Caulfried/Desktop/design_ran.csv')
design_ran_trans = (2*design_ran-1)/(2*30)###对各因素水平进行标准化处理
y0 = function(design_ran_trans['x1'],design_ran_trans['x2'])

###使用神经网络进行训练
import tensorflow as tf
tf.reset_default_graph()

n_inputs = 2
n_hidden1 = 15
n_hidden2 = 15
n_hidden3 = 15
n_hidden4 = 15
n_hidden5 = 10
n_output = 1

input_x = np.array(design_ran_trans)
input_y = np.array(y0)
dataset = tf.data.Dataset.from_tensor_slices((input_x,input_y))
dataset = dataset.batch(10).repeat()
iterator = dataset.make_one_shot_iterator()
batch_one_element = iterator.get_next() 

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(tf.int64, shape=(None), name='y')

from tensorflow.contrib.layers import fully_connected
with tf.name_scope('ann'):
    hidden1 = fully_connected(X, n_hidden1, scope='hidden1')
    hidden2 = fully_connected(hidden1, n_hidden2, scope='hidden2')
    hidden3 = fully_connected(hidden2, n_hidden3, scope='hidden3')
    hidden4 = fully_connected(hidden3, n_hidden4, scope='hidden4')
    hidden5 = fully_connected(hidden4, n_hidden5, scope='hidden5')
    y_ = fully_connected(hidden5, n_output, scope='output', activation_fn=None)

with tf.name_scope("loss"):
    mse = tf.reduce_mean(tf.square(y_-input_y))

learning_rate = 0.01
with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()
saver = tf.train.Saver()    

n_epochs = 100
batch_size = 10


with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(3):
            X_batch, y_batch = sess.run(batch_one_element)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
    accuracy_train = tf.reduce_mean(tf.square(y_-input_y))
    acc_train = accuracy_train.eval(feed_dict={X: input_x, y: input_y})
    print('Train_mse:',acc_train)###训练均方误差
    save_path = saver.save(sess, "D:/学习/研一上/tensorflow/my_model_final.ckpt")



with tf.Session() as sess:
    saver.restore(sess, "D:/学习/研一上/tensorflow/my_model_final.ckpt")
    X_test = np.array(Xtest)
    y_test = np.array(ytest)
    y_pred = y_.eval(feed_dict={X: X_test})
    accuracy = tf.reduce_mean(tf.square(y_pred-y_test))
    acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
    print('Test_mse:',acc_test)###测试均方误差
