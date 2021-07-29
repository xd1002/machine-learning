# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 09:32:34 2021

@author: Lenovo
"""

#矩阵求逆
import numpy as np
import copy
class matrix_inversion():
    def __init__(self,x):
        self.x=np.asarray(x,dtype=float)
        self.change_row=0
        self.change_column=0
    
    #change_element和calculate_determinant是两个有关计算行列式的程序
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
                    
                    #记录换过来的元素所在的行列，如果有一个与location相同则要在行列式中乘-1
                    self.change_row=i
                    self.change_column=j
                    return
          
        
    def calculate_determinant(self,x):
        
        #x需要转换为numpy中的array
        #x不需要换维数：由于偏置b的存在，需要在原数据加上全为1的一列
        
        
        #若为0阵，不用算就是0
        if np.all(x==0):
            return 0
        
        else:
            #记录计算行列式时为保持对角元素非0换行换列导致的行列式符号变化
            count=0
        #通过初等行变换将矩阵化为上三角矩阵
            for i in range(x.shape[0]-1):
                
                #首先将所在行的对角线元素换为非0元素
                self.change_element(x,i)
                if self.change_row==i or self.change_column==i:
                    if self.change_column!=self.change_row:
                       count=count+1 
                self.change_row=0
                self.change_column=0
                #初等行变换
                for j in range(i+1,x.shape[0]):
                    x[j,:]=x[j,:]-x[i,:]*x[j,i]/x[i,i]
                  
        #计算行列式(对角线元素乘积)
            k=1
            for i in range(x.shape[0]):
                k=k*x[i,i]
            return k*(-1)**count
 

    
    #下面才是矩阵求逆的主体部分
        
    def algebraic_cofactor(self,x,i,j):
        #用于将对应位置的代数余子式提出来        
        t=np.delete(x,i,0)
        t=np.delete(t,j,1)
        return ((-1)**(i+j))*self.calculate_determinant(t)
    
    
    def calculate_inversion(self):
        dimension=self.x.shape[0]
        self.inver=np.zeros((dimension,dimension))
        
        #如果不用copy的话无论if判断对错都会先计算一边行列式，即将矩阵化为上三角
        if self.calculate_determinant(copy.deepcopy(self.x))==0:
            print('The determinant of this matrix is 0,its inverse matrix doesn\'t exist')
        else:
            for i in range(dimension):
                for j in range(dimension):
                    self.inver[j,i]=self.algebraic_cofactor(self.x,i,j)
                    
            self.inver=self.inver/self.calculate_determinant(self.x)
            return self.inver








