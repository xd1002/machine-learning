# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 10:01:57 2021

@author: Lenovo
"""


import numpy as np
import copy

class QR_algorithm():
    def __init__(self,x):
        self.x=x


    def householder(self,x,indice):
        y=np.zeros(x.shape[0])
        y[indice]=1
        x_length=np.dot(x,x)**0.5
        y=x_length*y/len(indice)**0.5
        
        w=(x-y)/np.dot(x-y,x-y)**0.5
        #return w.shape
        return np.eye(x.shape[0])-2*np.matmul(w.reshape(-1,1),w.reshape(1,-1))
        
        
    def upper_hessenberg(self,x):
        indice=[0]
        for i in range(1,x.shape[0]-1):
            H=np.eye(x.shape[0])
            h=self.householder(x[i:,i-1],indice)
            
            H[i:,i:]=h
            x=np.matmul(H,x)
            x=np.matmul(x,H)
            
        return x
    
    def QR_decomposition(self,x,normalization=True):
        Q=np.zeros((x.shape[0],x.shape[0]))
        R=np.zeros((x.shape[0],x.shape[0]))+np.eye(x.shape[0])
        column_length=np.zeros(x.shape[0])
        
        if normalization==False:
            
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
    





    #epsilon决定了在数据的绝对值小于多少时认为他为0（类似与计算精度？）个人感觉10的-10次方好一点，这个值太大了太小了都会影响精度，尤其是太小了（如10的-20次方），会出现严重的计算错误。但10的-5次方等的情况下会出现精度不足的情况
    #建议尝试多个精度后取平均
    def eigvalue(self,x,max_iter_time,epsilon):
        x=self.upper_hessenberg(x)
        
        n=x.shape[0]-1
        s=x[n,n]
        iter_time=0
        self.eig_value=[]
        while n>=1:
            if np.max(np.abs(x[n,0:n]))>=epsilon:
                
                if iter_time>=max_iter_time:
                    a=x[n-1,n-1]
                    b=x[n-1,n]
                    c=x[n,n-1]
                    d=x[n,n]
                    
                    #这里一定要加float，不加不知道为什么会溢出（开方里面虽然是有限数，且不会趋近于0，但不知道为什么开方后就会溢出）
                    value1=(a+d+(float((a+d)**2-4*(a*d-b*c)))**0.5)/2
                    value2=(a+d-(float((a+d)**2-4*(a*d-b*c)))**0.5)/2
                    self.eig_value.append(value1)
                    self.eig_value.append(value2)
                    iter_time=0
                    n=n-2
                    x=x[0:n+1,0:n+1]
                    s=x[n,n]
                
                    
                else:
                    Q,R=self.QR_decomposition(x-s*np.eye(x.shape[0]))
                    x=np.matmul(R,Q)+s*np.eye(x.shape[0])
                    iter_time+=1
                    s=x[n,n]
                    
            else:
                self.eig_value.append(s)
                iter_time=0
                n-=1
                x=x[:n+1,:n+1]
                s=x[n,n]
                
        
                
                
        self.eig_value.append(x[0,0])
                
        return self.eig_value
    
    
    def change_element(self,x,location):
            
        for i in range(location,x.shape[0]):
            for j in range(location,x.shape[1]):
                if x[i,j]>=10**(-5):
                        
                    t=copy.deepcopy(x[location,:])
                    x[location,:]=x[i,:]
                    x[i,:]=t
                    return
                
                
    #计算行列式，如果行列式非0则直接输出解为0
    def calculate_determinant(self,x):
                
        #通过初等行变换将矩阵化为上三角矩阵
        for i in range(x.shape[0]-1):
            if (x[i+1:,i+1:]<=10**(-5)).all()==False:    
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
                A=np.asarray(A,dtype=np.complex)
                A_average=np.sum(A,axis=1)
                A_average=np.sum(A_average)
                A=A/np.complex(A_average)
                
            #if np.abs(self.calculate_determinant(A))>=10**(-5):
             #   print('end end end')
              #  return np.zeros(A.shape[0])
            #
            #else:
                
                #上面if的判断中已经算过了不用再算一边了
                for i in range(A.shape[0]-1):
                    if (np.abs(A[i+1:,i+1:])<=10**(-5)).all()==False:    
                        self.change_element(A,i)
                    else:
                        break
                    for j in range(i+1,A.shape[0]):
                        A[j,:]=A[j,:]-A[i,:]*A[j,i]/A[i,i]
                
                        
                
                for i in range(1,A.shape[0]):
                    if (np.abs(A[i,:])<=10**(-5)).all()==False:
                        for k in range(i,A.shape[0]):
                            if np.abs(A[i,k])>=10*(-5):
                                
                                for j in range(i):
                                    A[j,:]-=A[j,k]*A[i,:]/A[i,k]
                                
                                break
                            
                    else:
                        break
                
                index=[]
            
                for i in range(A.shape[0]):
                    if (np.abs(A[i,:])<=10**(-5)).all()==False:
                        for j in range(i,A.shape[0]):
                            if np.abs(A[i,j])>=10**(-5):
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
                self.basic_solution=np.delete(np.eye(A.shape[0],dtype=np.complex),index,1)
                self.basic_solution+=solution
                self.basic_solution=self.basic_solution.reshape(A.shape[0],-1)
                
                #return np.sum(self.basic_solution,axis=1)
            
            
    def calculate_eig_vector(self,x,eig_value):
        self.eig_vector=np.zeros((x.shape[0],len(eig_value)),dtype=np.complex)
        for i in range(len(eig_value)):
            self.solve(x-eig_value[i]*np.eye(x.shape[0]))
            if i!=0 and np.abs(eig_value[i]-eig_value[i-1])<=10**(-10):
                self.eig_vector[:,i]=(i+1)*self.basic_solution[:,0]+(i+2)*self.basic_solution[:,0]
            else:
                self.eig_vector[:,i]=(i+1)*self.basic_solution[:,0]
        return self.eig_vector

    
