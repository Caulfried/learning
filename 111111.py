# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 14:01:00 2019

@author: Caulfried
"""

import tensorflow as tf
import pandas as pd


input_x = np.array(design_ran_trans)
input_y = np.array(y0)

feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(input_x)
dnn_clf = tf.contrib.learn.DNNRegressor(hidden_units=[15,10],
                                         feature_columns=feature_columns)
dnn_clf.fit(x=input_x, y=input_y, batch_size=10, steps=100) 

from sklearn.metrics import mean_squared_error
y_pre_train = list(dnn_clf.predict(input_x))
mean_squared_error(input_y, y_pre_train)

y_pred = list(dnn_clf.predict(X_test)) 
mean_squared_error(y_test,y_pred)

##输出预测
out1 = pd.DataFrame(y_test,y_pred)
out1.to_csv('C:/Users/Caulfried/Desktop/out1.csv', index=True)

out2 = pd.DataFrame(y_test,y_pred)
out2.to_csv('C:/Users/Caulfried/Desktop/out2.csv', index=True)
