# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 11:21:14 2021

@author: Lenovo
"""

#解齐次线性方程组
#仅解变量数等于方程个数的齐次线性方程组
#使用初等行变换将矩阵变为行阶梯矩阵进行求解
#方程组的数值解法，LU分解感觉和这个差不多，高斯赛德尔迭代法和雅可比迭代法都要求矩阵为严格对角占优，条件较为苛刻

import copy
import numpy as np
class solve_homogeneous_linear_equation():
    
    #输入x：方程组的系数矩阵
    def __init__(self,x):
        self.x=x
    

    #必要时将每行的对角线元素换为非0元素    
    def change_element(self,x,location):
            
        for i in range(location,x.shape[0]):
            for j in range(location,x.shape[1]):
                if x[i,j]!=0:
                        
                    t=copy.deepcopy(x[location,:])
                    x[location,:]=x[i,:]
                    x[i,:]=t
                    return
                
                
    #计算行列式，如果行列式非0则直接输出解为0
    def calculate_determinant(self,x):
                
        #通过初等行变换将矩阵化为上三角矩阵
        for i in range(x.shape[0]-1):
            if (x[i+1:,i+1:]==0).all()==False:    
                self.change_element(x,i)
            else:
                break
                #初等行变换
            for j in range(i+1,x.shape[0]):
                x[j,:]=x[j,:]-x[i,:]*x[j,i]/x[i,i]
                  
        #计算行列式(对角线元素乘积)
        k=1
        for i in range(x.shape[0]):
            k=k*x[i,i]
        return k

              
    
    #求出基础解系，并默认各基向量间以系数1相加输出一个解        
    def solve(self,x):
            A=copy.deepcopy(x)
            A=np.asarray(A,dtype=float)
            
            if self.calculate_determinant(A)!=0:
                return np.zeros(A.shape[0])
            
            else:
                
                #上面if的判断中已经算过了不用再算一边了
                
                #for i in range(A.shape[0]-1):
                 #   if (A[i+1:,i+1:]==0).all()==False:    
                  #      self.change_element(A,i)
                   # else:
                    #    break
                    #for j in range(i+1,A.shape[0]):
                     #   A[j,:]=A[j,:]-A[i,:]*A[j,i]/A[i,i]
                        
                for i in range(1,A.shape[0]):
                    if (A[i,:]==0).all()==False:
                        for k in range(i,A.shape[0]):
                            if A[i,k]!=0:
                                
                                for j in range(i):
                                    A[j,:]-=A[j,k]*A[i,:]/A[i,k]
                                break
                            
                    else:
                        break
                
                index=[]
                for i in range(A.shape[0]):
                    if (A[i,:]==0).all()==False:
                        for j in range(i,A.shape[0]):
                            if A[i,j]!=0:
                                A[i,:]=A[i,:]/A[i,j]
                                index.append(j)
                                break
                    else:
                        break
                
                n=len(index)
                for i in range(n):
                    if index[n-1-i]!=n-1-i:
                        A[index[n-1-i],:]=copy.deepcopy(A[n-1-i,:])
                        A[n-1-i,:]=0
                                
                
                solution=-np.delete(copy.deepcopy(A),index,1)
                self.basic_solution=np.delete(np.eye(A.shape[0]),index,1)
                self.basic_solution+=solution
                
                return np.sum(self.basic_solution,axis=1)

