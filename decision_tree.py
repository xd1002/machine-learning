# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 08:51:09 2021

@author: Lenovo
"""

#决策树
#特征全为离散特征
import pandas as pd
import numpy as np
import copy
import math

class decision_tree():
    
    def __init__(self,num_a):
        self.num_a=num_a#每个属性的可能取值数
        self.tree=[]#记录当前构建的树
        self.p=0#指针，从0开始，指向当前处理的位置
        
        #想法：用字典表示树的每个节点，具体形式：{‘特征’:[],'值'：[],'类别':[],'左孩子'：[],'右孩子':[]}
        
    def fit(self,x,y,a):
        #x：数据，pd.DataFrame格式
        #y：数据对应标签，二分类，-1和1
        self.tree.append({'特征':[],'值':[],'类别':[],'孩子':[]})
        #看本次子集中样本类别是否相同，并记录同类别样本的个数
        count=0
        y=pd.DataFrame(y)
        
        #如果集合中只有一个样本
        if x.shape[0]==1:
            self.tree[self.p]['类别']=y[0]
        else:
            
        
            #记录当前样本集中正类个数
            for i in list(y.index):
                if y[y.columns[0]][i]==y[y.columns[0]][list(y.index)[0]]:
                    count+=1
                    
            #如果集合中所有样本为一类
            if count==x.shape[0]:
                self.tree[self.p]['类别']=y[0]
            else:
                
            
                #如果特征集为空集，则返回比率最大的一个类
                if len(a)==0:
                    if count>=x.shape[0]-count:
                        self.tree[self.p]['类别']=y[0]
                        
                    else:
                        self.tree[self.p]['类别']=-y[0]
                
                else:
                    
                        
        
                    indice_num_dict=self.find_possible_num(x,y,a)
                        
                    #找到在当前特征集与样本集下的最优分割特征
                    a_best=self.find_best_a(x,y,a,indice_num_dict)
                    self.tree[self.p]['特征'].append(a_best)
                    
                    self.tree[self.p]['值'].append(indice_num_dict[6][a_best])
                    self.tree[self.p]['值']=sum(self.tree[self.p]['值'],[])
                    
                    p_tree=copy.deepcopy(self.p)
                    
                    
                    
                    #遍历每一个可能取值
                    for i in range(len(indice_num_dict[6][a_best])):
                        self.p+=1
                        self.tree[p_tree]['孩子'].append(self.p)
                        if np.all(np.array(y[y.columns[0]][indice_num_dict[8][a_best][i]])==1) or np.all(np.array(y[y.columns[0]][indice_num_dict[8][a_best][i]])==-1):
                            self.tree.append({'特征':[a_best],'值':[indice_num_dict[6][a_best][i]],'类别':[y[y.columns[0]][indice_num_dict[8][a_best][i][0]]],'孩子':[]})
                        else:
                            a_modify=list(copy.deepcopy(a))
                            a_modify.remove(a_best)
                            row_indice=indice_num_dict[8][a_best][i]
                            data_x=copy.deepcopy(x.loc[row_indice,a_modify])
                            data_y=copy.deepcopy(y[y.columns[0]][row_indice])
                            self.fit(data_x,data_y,a_modify)
                            
                
                
            
            
            
        #找数据集x中在特征a中所有特征的可能取值，及数据集所有可能取值在不同类别（正类、负类）对应出现个数与indice    
    def find_possible_num(self,x,y,a):
        
        #indice_num_dict用于存储上述说明中涉及到的
        #第0个：存储正类所有可能取值
        #第1个：存储负类所有可能取值
        #第2个：存储正类中可能取值的个数
        #第3个：存储正类中可能取值对应的indice
        #第4个：存储负类中可能的取值个数
        #第5个：存储负类中可能取值对应的indice
        #第6个：存储所有类别的可能取值
        #第7个：存储所有类别可能取值的个数
        #第8个：存储所有类别可能取值对应的indice
        
        #为了编写方便，不单独生成5个dict，使用list进行批量操作
        y=pd.DataFrame(y)
        
        #有的时候y会是series格式，为防止麻烦直接进行修改
        y_col_name=y.columns
        indice_num_dict=[{},{},{},{},{},{},{},{},{}] 
        
        
        #初始化每个dict（将特征集a中的每个元素作为key置入dict中，并使其对应值为空list[]）
        for i in range(9):
            for j in range(len(a)):
                indice_num_dict[i][a[j]]=[]
                
        
        data_x=copy.deepcopy(x[a])
        data_y=copy.deepcopy(y)
        
        
        #遍历数据的每列（每个特征）
        for i in range(data_x.shape[1]):
            
            #遍历数据的每行(每个样本)
            for j in list(x.index):
                
                #如果全部可能取值中无，则在单独可能取值中也无，需要创建
                if data_x[a[i]][j] not in indice_num_dict[6][a[i]]:
                    
                    #在正负类的dict中创建
                    if data_y[y_col_name[0]][j]==1: 
                        indice_num_dict[0][a[i]].append(data_x[a[i]][j])
                        indice_num_dict[2][a[i]].append(1)
                        indice_num_dict[3][a[i]].append([j])
                    else:
                        indice_num_dict[1][a[i]].append(data_x[a[i]][j])
                        indice_num_dict[4][a[i]].append(1)
                        indice_num_dict[5][a[i]].append([j])
                    
                    #在所有类的dict中创建
                    indice_num_dict[6][a[i]].append(data_x[a[i]][j])
                    indice_num_dict[7][a[i]].append(1)
                    indice_num_dict[8][a[i]].append([j])
                    
                        
                        
                #如果在全部可能取值中有，则不在1中就在-1中
                else:
                    
                    #在正负类的dict中进行添加
                    if data_y[y_col_name[0]][j]==1:   
                        
                        if data_x[a[i]][j] not in indice_num_dict[0][a[i]]:
                            indice_num_dict[0][a[i]].append(data_x[a[i]][j])
                            indice_num_dict[2][a[i]].append(1)
                            indice_num_dict[3][a[i]].append([j])
    
                        else:
                            indice=indice_num_dict[0][a[i]].index(data_x[a[i]][j])
                            indice_num_dict[2][a[i]][indice]+=1
                            indice_num_dict[3][a[i]][indice].append(j)
                            
                            
                    else:
                        if data_x[a[i]][j] not in indice_num_dict[1][a[i]]:
                            indice_num_dict[1][a[i]].append(data_x[a[i]][j])
                            indice_num_dict[4][a[i]].append(1)
                            indice_num_dict[5][a[i]].append([j])
    
                        else:
                            indice=indice_num_dict[1][a[i]].index(data_x[a[i]][j])
                            indice_num_dict[4][a[i]][indice]+=1
                            indice_num_dict[5][a[i]][indice].append(j)
                    
                    #在全部类的dict中进行添加
                    indice=indice_num_dict[6][a[i]].index(data_x[a[i]][j])
                    indice_num_dict[7][a[i]][indice]+=1
                    indice_num_dict[8][a[i]][indice].append(j)
                    
    
                        
                    #若为正类，记录可能取值，取值个数与取值indice
                                           
                    
                    #若为负类，记录可能取值，取值个数与取值的indice
        return indice_num_dict
            
    
    #特征选取，计算经验熵
    def empirical_entropy(self,x,y):
        entropy=0
        for i in [1,-1]:
            if y[y[y.columns]==i].empty:
                entropy+=0
            else:
                rate=y[y[y.columns]==i].shape[0]/y.shape[0]
                entropy-=rate*math.log(rate,2)
        return entropy
    
    
    #特征选取，计算关于特征a经验条件熵,a就是一个字符串
    def empirical_conditional_entropy(self,x,y,a,indice_num_dict):
        conditional_entropy=0
        D=x.shape[0]
        for i in indice_num_dict[6][a]:
            indice=indice_num_dict[6][a].index(i)
            Di=indice_num_dict[7][a][indice]
            if i in indice_num_dict[0][a]:
                indice1=indice_num_dict[0][a].index(i)
                Di1=indice_num_dict[2][a][indice1]
                conditional_entropy-=Di/D*Di1/Di*math.log(Di1/Di,2)
            if i in indice_num_dict[1][a]:
                indice0=indice_num_dict[1][a].index(i)
                Di0=indice_num_dict[4][a][indice0]
                conditional_entropy-=Di/D*Di0/Di*math.log(Di0/Di,2)
            
        return conditional_entropy
    
    
    #找最优划分特征
    def find_best_a(self,x,y,a,indice_num_dict):
        empirical_entropy=self.empirical_entropy(x,y)
        best_a=a[0]
        max_information_increase=empirical_entropy-self.empirical_conditional_entropy(x,y,best_a,indice_num_dict)
        for i in a:
            information_increase=empirical_entropy-self.empirical_conditional_entropy(x,y,i,indice_num_dict)
            if information_increase>max_information_increase:
                best_a=i
                max_information_increase=information_increase
        return best_a
    
    
    def predict(self,x,y):
        
        predict_list=[]
        for i in list(x.index): 
            p=0#指针
            while self.tree[p]['类别']==[]:        
                indice=self.tree[p]['值'].index(x[self.tree[p]['特征'][0]][i])
                p=self.tree[p]['孩子'][indice]
            
            predict_list.append(self.tree[p]['类别'][0])
            
        
        right_num=0
        for i in list(y.index):
            if y[y.columns[0]][i]==predict_list[list(y.index).index(i)]:
                right_num+=1
                
        print('预测正确个数为',right_num,'\n','预测正确率为',right_num/x.shape[0])
                
                

