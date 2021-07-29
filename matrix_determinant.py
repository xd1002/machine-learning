# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 20:59:10 2021

@author: Lenovo
"""

#矩阵求逆（初等行列变化）

import numpy as np
import copy

class matrix_determinant():
    def __init__(self,x):
        self.x=np.asarray(x,dtype=float)

        
    def change_element(self,x,location):
        
        #先将非0元素换到矩阵（0，0）处
        #location代表当前研究位于第几行的对角线元素（该行数表示array中的行数编号，及从0开始）
        for i in range(location,x.shape[0]):
            for j in range(location,x.shape[1]):
                if x[i,j]!=0:
                    
                    #先交换行
                    #不可通过直接赋值进行转换，直接赋值赋的是指针
                    t=copy.deepcopy(x[location,:])
                    x[location,:]=x[i,:]
                    x[i,:]=t
                    #再交换列
                    t=copy.deepcopy(x[:,location])
                    x[:,location]=x[:,j]
                    x[:,j]=t
                    return
          
        
    def calculate_determinant(self):
        
        #x需要转换为numpy中的array
        #x不需要换维数：由于偏置b的存在，需要在原数据加上全为1的一列
        
        
        #若为0阵，不用算就是0
        if np.all(self.x==0):
            return 0
        
        else:
        #通过初等行变换将矩阵化为上三角矩阵
            for i in range(self.x.shape[0]-1):
                
                #首先将所在行的对角线元素换为非0元素
                self.change_element(self.x,i)
                
                #初等行变换
                for j in range(i+1,self.x.shape[0]):
                    self.x[j,:]=self.x[j,:]-self.x[i,:]*self.x[j,i]/self.x[i,i]
                  
        #计算行列式(对角线元素乘积)
            k=1
            for i in range(self.x.shape[0]):
                k=k*self.x[i,i]
            return k
                


