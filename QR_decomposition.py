# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 21:26:29 2021

@author: Lenovo
"""

import numpy as np

#QR分解
#使用施密特正交化(Gram_Schmidt)
#x:方阵，请自行将该方阵处理为满秩
#normalization:   True：对组成Q的正交向量进行归一化   False：不进行归一化
#self.decomposition输出正交矩阵Q和上三角矩阵R

class QR_decomposition():
    def __init__(self,x,normalization=True):
        self.x=x
        self.normalization=normalization
        
    def decomposition(self,x):
        Q=np.zeros((x.shape[0],x.shape[0]))
        R=np.zeros((x.shape[0],x.shape[0]))+np.eye(x.shape[0])
        column_length=np.zeros(x.shape[0])
        
        if self.normalization==False:
            
            for i in range(x.shape[0]):
                Q[:,i]=x[:,i]
                
                for j in range(i):    
                    R[j,i]=np.dot(x[:,i],Q[:,j])/np.dot(Q[:,j],Q[:,j])
                    Q[:,i]-=R[j,i]*Q[:,j]
                    
        else:
            
            for i in range(x.shape[0]):
                Q[:,i]=x[:,i]
                
                for j in range(i):    
                    Q[:,i]-=np.dot(x[:,i],Q[:,j])/np.dot(Q[:,j],Q[:,j])*Q[:,j]
                column_length[i]=np.dot(Q[:,i],Q[:,i])**0.5
            
            Q=Q/column_length
            
            for i in range(x.shape[0]):
                for j in range(i+1):
                    R[j,i]=np.dot(Q[:,j],x[:,i])
                    
        return Q,R


