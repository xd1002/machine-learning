# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 08:55:41 2021

@author: Lenovo
"""

import numpy as np

#householder变换，不作过多介绍，自行百度
#x:输入向量，numpy.array([x,x,x……])形式
#indice:不用置0的向量分量
#self.calculate_H:输入向量x，输出能将其除了indice外其余下标都置0的变换矩阵9（即Hx除了indice外其余全部置0）


class householder():
    def __init__(self,x,indice):
        self.x=x
        self.indice=indice
        
    def calculate_H(self,x):
        y=np.zeros(x.shape[0])
        y[self.indice]=1
        x_length=np.dot(x,x)**0.5
        y=x_length*y/len(self.indice)**0.5
        
        w=(x-y)/np.dot(x-y,x-y)**0.5
        
        return np.eye(x.shape[0])-2*np.matmul(w.reshape(-1,1),w.reshape(1,-1))



