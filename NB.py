# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 15:25:17 2021

@author: Lenovo
"""

import numpy as np
import copy


#朴素贝叶斯
#这次使用pandas的DataFrame形式
#数据每一列的数据类型一定要明确，不能产生int编程float的情况
#文字类型数据一定要转化为数字类型的数据

class NB():
    
    def __init__(self,possible_num):
        self.possible_num=possible_num#记录每种特征的可能取值数，如果是连续值就记可能取值数为0，用list存储
        
        
        
    
    #生成四个字典，其中两个存放正类和负类中每个离散特征的不同类别，用list存储，如果是连续特征则是空list
    #另两个字典存放正类和负类每个离散特征不同类别的出现频率，用list存储，如果是连续特征则是空list
    
    #x：输入数据，一定要是pd.DataFrame类型的
    #y：每个样本对应的类别(仅支持二分类，类别为1和-1（正类与负类）)
    def create_four_dict(self,x,y):
        
        
        #首先是正类和负类的特征不同类别
        self.positive_dict_0={}
        self.negative_dict_0={}
        
        
        #其次是正类和负类的特征不同类别出现频率
        self.positive_dict_1={}
        self.negative_dict_1={}
        
        
        #创建一个dataframe x的复制，并改变行名和列明为数组的命名方式（0，1，2，3……）
        data_x=copy.deepcopy(x)
        data_y=copy.deepcopy(y)
        
        
        #记录数据x和y的原本特征名
        self.x_real_name=list(x.columns)
        self.y_real_name=list(y.columns)
        
        
        #改变data_x和data_y的行名与列名
        data_x_new_column_name=[]
        data_x_new_index_name=[]
        for i in range(x.shape[1]):
            data_x_new_column_name.append(i)
        for j in range(x.shape[0]):
            data_x_new_index_name.append(j)
        data_x.columns=data_x_new_column_name
        data_x.index=data_x_new_index_name
        data_y.columns=[0]
        data_y.index=data_x_new_index_name
        #下面全部使用data_x和data_y
        
        #下面生成上面创建的字典
        #首先在每个字典中生成对应的key
        for i in range(data_x.shape[1]):
                self.negative_dict_0[self.x_real_name[i]]=[]
                self.negative_dict_1[self.x_real_name[i]]=[]
                self.positive_dict_0[self.x_real_name[i]]=[]
                self.positive_dict_1[self.x_real_name[i]]=[]
                
        
        positive_num=0
        positive_indice=[]
        negative_num=0
        negative_indice=[]
        for i in range(data_x.shape[0]):
            if data_y[0][i]==-1:
                negative_num+=1
                negative_indice.append(i)
            else:
                positive_num+=1
                positive_indice.append(i)
                
                
        #生成存储不同特征不同类别的字典
        for i in range(data_x.shape[1]):
            #possible_num记录特征的可能取值，如果为0就是连续型的
            if self.possible_num[i]!=0:
                for j in range(data_x.shape[0]):
                    if data_y[0][j]==-1:
                        if data_x[i][j] not in self.negative_dict_0[self.x_real_name[i]]:
                            self.negative_dict_0[self.x_real_name[i]].append(data_x[i][j])
                            self.negative_dict_1[self.x_real_name[i]].append(2/(negative_num+self.possible_num[i]))
                            continue
                        else:
                            indice=self.negative_dict_0[self.x_real_name[i]].index(data_x[i][j])
                            self.negative_dict_1[self.x_real_name[i]][indice]+=1/(negative_num+self.possible_num[i])
                            
                            
                    if data_y[0][j]==1:
                        if data_x[i][j] not in self.positive_dict_0[self.x_real_name[i]]:
                            self.positive_dict_0[self.x_real_name[i]].append(data_x[i][j])
                            self.positive_dict_1[self.x_real_name[i]].append(2/(positive_num+self.possible_num[i]))
                            continue
                        else:
                            indice=self.positive_dict_0[self.x_real_name[i]].index(data_x[i][j])
                            self.positive_dict_1[self.x_real_name[i]][indice]+=1/(positive_num+self.possible_num[i])
                            
            else:
                #如果是连续型特征，对应存储特征类别频率的列表存储对应数据的样本均值（第一位）和样本方差（第二位）
                average_negative=data_x[i][negative_indice].sum()/negative_num
                average_positive=data_x[i][positive_indice].sum()/positive_num
                square_error_negative=((data_x[i][negative_indice]-average_negative)**2).sum()/negative_num
                square_error_positive=((data_x[i][positive_indice]-average_positive)**2).sum()/positive_num
                
                self.negative_dict_1[self.x_real_name[i]].append(average_negative)
                self.positive_dict_1[self.x_real_name[i]].append(average_positive)
                
                self.negative_dict_1[self.x_real_name[i]].append(square_error_negative)
                self.positive_dict_1[self.x_real_name[i]].append(square_error_positive)

                        
            
            
    def predict(self,x,y):
        self.positive_probability=[]
        self.negative_probability=[]
        
        #创建一个dataframe x的复制，并改变行名和列明为数组的命名方式（0，1，2，3……）
        data_x=copy.deepcopy(x)
        data_y=copy.deepcopy(y)
        
        #改变data_x和data_y的行名与列名
        data_x_new_column_name=[]
        data_x_new_index_name=[]
        for i in range(x.shape[1]):
            data_x_new_column_name.append(i)
        for j in range(x.shape[0]):
            data_x_new_index_name.append(j)
        data_x.columns=data_x_new_column_name
        data_x.index=data_x_new_index_name
        data_y.columns=[0]
        data_y.index=data_x_new_index_name
        
        
        positive_num=0
        positive_indice=[]
        negative_num=0
        negative_indice=[]
        for i in range(data_x.shape[0]):
            if data_y[0][i]==-1:
                negative_num+=1
                negative_indice.append(i)
            else:
                positive_num+=1
                positive_indice.append(i)


        for i in range (data_x.shape[0]):
            pro_plus=1
            pro_minus=1
            for j in range(data_x.shape[1]):
                if self.negative_dict_0[self.x_real_name[j]]!=[] or self.positive_dict_0[self.x_real_name[j]]!=[]:
                    
                    
                    if data_x[j][i] in self.negative_dict_0[self.x_real_name[j]]:
                        indice_minus=self.negative_dict_0[self.x_real_name[j]].index(data_x[j][i])
                        pro_minus=pro_minus*self.negative_dict_1[self.x_real_name[j]][indice_minus]
                    else:
                        pro_minus=pro_minus*1/(negative_num+self.possible_num[j])
                        
                    
                    if data_x[j][i] in self.positive_dict_0[self.x_real_name[j]]:
                        indice_plus=self.positive_dict_0[self.x_real_name[j]].index(data_x[j][i])
                        pro_plus=pro_plus*self.positive_dict_1[self.x_real_name[j]][indice_plus]
                    else:
                        pro_plus=pro_plus*1/(positive_num+self.possible_num[j])
                        
                    
                        
                else:
                    plus_exp=-((data_x[j][i]-self.positive_dict_1[self.x_real_name[j]][0])**2)/2/self.positive_dict_1[self.x_real_name[j]][1]
                    minus_exp=-((data_x[j][i]-self.negative_dict_1[self.x_real_name[j]][0])**2)/2/self.negative_dict_1[self.x_real_name[j]][1]
                    pro_plus=pro_plus*np.exp(plus_exp)/(np.sqrt(2*np.pi*self.positive_dict_1[self.x_real_name[j]][1]))
                    pro_minus=pro_minus*np.exp(minus_exp)/(np.sqrt(2*np.pi*self.negative_dict_1[self.x_real_name[j]][1]))
            self.positive_probability.append(pro_plus*positive_num/data_x.shape[0])
            self.negative_probability.append(pro_minus*negative_num/data_x.shape[0])
        
        self.predict_result=[]
        for i in range(data_x.shape[0]):
            if self.positive_probability[i]>=self.negative_probability[i]:
                self.predict_result.append(1)
            else:
                self.predict_result.append(-1)
        
        
        right_num=0
        for i in range(data_x.shape[0]):
            if self.predict_result[i]==data_y[0][i]:
                right_num+=1
                
        print('预测正确个数：',right_num,'\n','预测正确率：',right_num/data_x.shape[0])


