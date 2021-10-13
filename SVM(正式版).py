# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 20:53:19 2021

@author: Lenovo
"""

import numpy as np
import random
import copy
import matplotlib.pyplot as plt


#SVM,针对数据线性可分的情况

class SVM():
    def __init__(self,x,
                 y,
                 max_iter_time,
                 punishment):
        #输入样本x（每行为一个数据样本）
        self.x=x
        #每个样本对应类别（-1，1）
        self.y=y
        self.max_iter_time=max_iter_time
        self.punishment=punishment#类似于软间隔支持向量机中的惩罚系数C
        
        
    def fit(self,x,y):
        
        
        #x：每行为一个样本
        x=x.T
        size_x=x.shape[1]#样本个数
        self.alpha=np.zeros(size_x)
        
                
        #alpha初始化为0，w和b也初始化为0
        self.w=np.zeros(x.shape[0])
        self.b=0
        
        
        #训话中需要用到的计数器
        change_times=0#记录alpha变化次数
        unchange_iter_time=0#记录alpha不变的次数
        self.final_iter_num=0#记录总的迭代次数
        
        
        #当alpha不变次数超过max_iter_time时停止迭代
        while unchange_iter_time<=self.max_iter_time:
            self.final_iter_num+=1
            change_times=0
            alpha1=[]
            alpha2=[]
            
            
            #记录所有不满足KKT条件的alpha的indice        
            for i in range(self.alpha.shape[0]):
                if self.alpha[i]<=10**(-20) and self.alpha[i]>=0 and y[i]*(np.dot(self.w,x[:,i])+self.b)<1:
                    alpha1.append(i)
                if self.alpha[i]==self.punishment and y[i]*(np.dot(self.w,x[:,i])+self.b)>1:
                    alpha1.append(i)
                if self.alpha[i]<self.punishment and self.alpha[i]>0 and y[i]*(np.dot(self.w,x[:,i])+self.b)!=1:
                    alpha1.append(i)
                
            
            #如果选不出来则停止迭代
            if alpha1==[]:
                print('满足KKT条件，是最优解')
                return self.w,self.b
            
    
            #如果不停止迭代则从中随机选择一个（之前每次使用第一个不满足KKT条件的alpha分量，非常不稳定）
            indice=int(random.uniform(0,len(alpha1)))
            alpha1=[alpha1[indice]]
            
            
            #随机选择一个和已经选定的alpha1不一样的alpha2（如果使用网上所说选择使得|E1-E2|最大的alpha2，也是极度不稳定）
            #不知道是不是numpy包运算造成的问题
            alpha2=alpha1    
            while alpha2==alpha1:
                alpha2=[int(random.uniform(0,self.alpha.shape[0]))]
                
                
                
                
            #diff_max=0#记录|E1-E2|的最大值
            #max_indice=0#记录当前最大值对应的下标
            #for j in range(self.alpha.shape[0]):
                #if j==alpha1[0]:
                    #continue
                #else:
                    #diff_now=abs(np.dot(w,x[:,alpha1[0]])-y[alpha1[0]]-np.dot(w,x[:,j])+y[j])
                    #if diff_now>diff_max:
                        #diff_max=diff_now
                        #max_indice=j
            #alpha2=[max_indice]

        
            alpha1_old=copy.deepcopy(self.alpha[alpha1[0]])
            alpha2_old=copy.deepcopy(self.alpha[alpha2[0]])
            if y[alpha1[0]]==y[alpha2[0]]:
                L=max(0,alpha1_old+alpha2_old-self.punishment)
                H=min(self.punishment,alpha1_old+alpha2_old)
            else:
                L=max(0,alpha2_old-alpha1_old)
                H=min(self.punishment,self.punishment+alpha2_old-alpha1_old)
                
            #计算不考虑L和H的约束（即未剪辑情况）时新alpha1和alpha2的值
            alpha2_new_unclip=alpha2_old+y[alpha2[0]]*(np.dot(self.w,x[:,alpha1[0]])-y[alpha1[0]]-np.dot(self.w,x[:,alpha2[0]])+y[alpha2[0]])/np.dot(x[:,alpha1[0]]-x[:,alpha2[0]],x[:,alpha1[0]]-x[:,alpha2[0]])
            if alpha2_new_unclip>H:
                alpha2_new=H
            else:
                if alpha2_new_unclip<L:
                    alpha2_new=L
                else:
                    alpha2_new=alpha2_new_unclip
            if abs(alpha2_new-alpha2_old)<=10**(-30):
                print('变化过小')
                unchange_iter_time+=1
                continue
            
            alpha1_new=alpha1_old+y[alpha1[0]]*y[alpha2[0]]*(alpha2_old-alpha2_new)
            self.alpha[alpha1[0]]=alpha1_new
            self.alpha[alpha2[0]]=alpha2_new
            self.w=np.sum(self.alpha*y*x,axis=1)
            b1_new=y[alpha1[0]]-np.dot(self.w,x[:,alpha1[0]])
            b2_new=y[alpha2[0]]-np.dot(self.w,x[:,alpha2[0]])
            self.b=(b1_new+b2_new)/2


            change_times+=1
            print("第%d次迭代 样本:%d, alpha优化次数:%d" % (unchange_iter_time,i,change_times))

            if change_times==0:
                unchange_iter_time+=1
                
            else:
                unchange_iter_time=0
        print('总迭代次数=',self.final_iter_num)
        
    
    #预测正确率
    
    def predict(self,x,y):
        x=x.T
        count=0
        for i in range(x.shape[1]):
            if y[i]*(np.dot(self.w,x[:,i])+self.b)>=0:
                count+=1
        return count,x.shape[1],count/x.shape[1]#输出正确个数，样本个数和正确率
    
    
    #只能画二维的图
    
    def plot_hyper_plane(self,data_x,data_y,min_data,max_data):
        plt.scatter(data_x[data_y==-1,0],data_x[data_y==-1,1],c='red')
        plt.scatter(data_x[data_y==1,0],data_x[data_y==1,1],c='blue')
        data=np.linspace(min_data,max_data)
        y_0=(-self.b-self.w[0]*data)/self.w[1]
        y_1_plus=(1-self.b-self.w[0]*data)/self.w[1]
        y_1_minus=(-1-self.b-self.w[0]*data)/self.w[1]
        plt.plot(data,y_0)
        plt.plot(data,y_1_plus,linestyle='--')
        plt.plot(data,y_1_minus,linestyle='--')
        plt.grid()
        
        plt.scatter(data_x[self.alpha!=0,0],data_x[self.alpha!=0,1],s=120,c='none',marker='o',edgecolor='black')

        
    
