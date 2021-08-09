# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 10:01:57 2021

@author: Lenovo
"""


import numpy as np


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
    
    
    def eigvalue(self,x,max_iter_time):
        x=self.upper_hessenberg(x)
        n=x.shape[0]-1
        s=x[n,n]
        epsilon=10**(-10)
        iter_time=0
        eig_value=[]
        while n>=1:
            if np.max(np.abs(x[n,0:n]))>=epsilon:
                
                if iter_time>=max_iter_time:
                    a=x[n-1,n-1]
                    b=x[n-1,n]
                    c=x[n,n-1]
                    d=x[n,n]
                    value1=(a+d+((a+d)**2-4*(a*d-b*c))**0.5)/2
                    value2=(a+d-((a+d)**2-4*(a*d-b*c))**0.5)/2                    
                    eig_value.append(value1)
                    eig_value.append(value2)
                    iter_time=0
                    n=n-2
                    
                else:
                    Q,R=self.QR_decomposition(x[:n+1,:n+1]-s*np.eye(n+1))
                    x=np.matmul(R,Q)+s*np.eye(n+1)
                    iter_time+=1
                    s=x[n,n]
                    
            else:
                eig_value.append(s)
                iter_time=0
                n-=1
                
                
        eig_value.append(x[0,0])
                
        return eig_value
                    

