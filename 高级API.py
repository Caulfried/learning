# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 22:50:47 2019

@author: Caulfried
"""

import tensorflow as tf

feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(input_x)
dnn_clf = tf.contrib.learn.DNNRegressor(hidden_units=[15,10],
                                         feature_columns=feature_columns)
dnn_clf.fit(x=input_x, y=input_y, batch_size=10, steps=100) 

from sklearn.metrics import mean_squared_error
y_pre_train = list(dnn_clf.predict(input_x))
mean_squared_error(input_y, y_pre_train)

y_pred = list(dnn_clf.predict(X_test)) 
mean_squared_error(y_test,y_pred)
