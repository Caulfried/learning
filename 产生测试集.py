# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 17:44:19 2019

@author: Caulfried
"""

###随机产生一个两因素设计
#import random
#import numpy as np
#import pandas as pd
#x1 = list(np.arange(1,31))
#x2 = list(np.arange(1,31))
#random.shuffle(x1)
#random.shuffle(x2)
#Xtest = pd.DataFrame({'x1':x1,'x2':x2})
#Xtest.to_csv('C:/Users/Caulfried/Desktop/Xtest.csv', index=False)

Xtest = pd.read_csv('C:/Users/Caulfried/Desktop/Xtest.csv')
Xtest = (2*Xtest-1)/(2*30)###对各因素水平进行标准化处理
ytest = function(Xtest['x1'],Xtest['x2'])
