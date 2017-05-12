# -*- coding: utf-8 -*-
from numpy import loadtxt
import random
import numpy as np
import matplotlib.pyplot as plt
import math
from numpy.linalg import inv
from scipy.stats import multivariate_normal
from numpy import linalg as LA

def cal_dist(x,y):
    dis=0.0
    if len(x) != len(y):
        print "Not same Col number"
        return
    for i in range(len(x)):
        dis =dis+ pow(x[i]-y[i],2)
    return math.sqrt(dis) 
    
def initial_Point(k,train_x):
    return train_x[random.sample(range(0,train_x.shape[0]),k)]
    
 #kMeans Algorithm 을 이용하여 gaussian bases mean와 variance를 구한다      
def kMeans(train_x,k):
    points=initial_Point(k,train_x)
    while True:
     class_k =[]
     class_k_mean=[]
     #class_k_var=[]
     for i in range(k):
        class_k.append([])             # class list make
        
     for i in range(len(train_x)):
        dist_sort=[]
        for j in range(len(points)):
            dist_sort.append(cal_dist(points[j],train_x[i]))
        min_dist=np.argmin(dist_sort)
        class_k[min_dist].append(i)                     # classification
        
     for i in range(k):
        class_k_mean.append([])
        #class_k_var.append([])
        for j in range(train_x.shape[1]):
            class_k_mean[i].append(np.mean(train_x[class_k[i],j]))  
            #class_k_var[i].append(np.var(train_x[class_k[i],j]))
     class_k_mean=np.array(class_k_mean) # mean nparray
     #class_k_var=np.array(class_k_var) #var nparray
  
     if (points == class_k_mean).all() == False:
        points = class_k_mean
        continue    
     else:
         break
    class_k_var =np.repeat([0.01],k)
    #class_k_var =estiVar(class_k_mean,class_k,train_x)     
    return class_k_mean,class_k_var   
    
def probDensity(mean,var,x):
    ex_val = -math.pow(x-mean,2)/(2*var)
    p = 1/(math.sqrt(2*var*math.pi))* math.exp(ex_val)
    return p
    
def probExp(mean,var,x):
    value=-pow(cal_dist(x,mean),2)/(2*var) 
    return math.exp(value)

def estiVar(mean,class_k,train_x):
    dist=[]
    var_temp=[]
    for i in range(mean.shape[0]):
        dist.append([])
        for j in range(len(class_k[i])):
            dist[i].append(cal_dist(mean[i],train_x[class_k[i][j]]))
        var_temp.append(np.mean(dist[i]))
    var_temp=np.array(var_temp)
    return var_temp    
    
    
#RBF Weight
def rbf_kMeans_W(train_x,train_y,k,mean,var):
    if (var==0).any() == True:
       var[np.where(var==0)]=1
    
    fi=[]
    for j in range(train_x.shape[0]):
      fi.append([])
      for i in range(k):
          fi[j].append(probExp(mean[i],var[i],train_x[j]))
    fi=np.array(fi)
    fi=np.c_[fi,np.repeat([1],train_x.shape[0])]
    #print "fi shape:",fi.shape    
        
    
    weight=np.dot(np.dot(inv(np.dot(np.transpose(fi),fi)+0.01*np.eye(fi.shape[1])),np.transpose(fi)),train_y)
    print "\nweight shape",weight.shape
    
    #esti_y= np.dot(fi,weight)
    #print "esti_y",esti_y.shape
    
    return weight

 #RBF Network     
def rbf_kMeans_Fi(train_x,k,mean,var):
    if (var==0).any() == True:
       var[np.where(var==0)]=1
    
    fi=[]
    for j in range(train_x.shape[0]):
      fi.append([])
      for i in range(k):
          fi[j].append(probExp(mean[i],var[i],train_x[j]))
    fi=np.array(fi)
    fi=np.c_[fi,np.repeat([1],train_x.shape[0])]
    #print "fi shape: ", fi.shape
    
    return fi
    
def MSE(esti_y,train_y):
    SUM=0.0
    for i in range(train_y.shape[0]):
        temp=pow((esti_y-train_y)[i],2)
        SUM+=temp
    return SUM/train_y.shape[0]
                              
if __name__ == '__main__':
#load text data
   fa_train1= loadtxt("fa_train1.txt",delimiter="\t",unpack=False)
   fa_train2= loadtxt("fa_train2.txt",delimiter="\t",unpack=False)
   fa_test= loadtxt("fa_test.txt",delimiter="\t",unpack=False)

   print "fa_train1: ",fa_train1.shape
   print "fa_train2: ",fa_train2.shape
   print "fa_test: ",fa_test.shape


   train1_x,train1_y = fa_train1[:,:1],fa_train1[:,1]
   train2_x,train2_y = fa_train2[:,:1],fa_train2[:,1]
   test_x,test_y = fa_test[:,:1],fa_test[:,1]
  
   ####cis_train1
   k1=15
   mean1, var1 = kMeans(train1_x,k1)
   weight1=rbf_kMeans_W(train1_x,train1_y,k1,mean1,var1)
   fi1=rbf_kMeans_Fi(train1_x,k1,mean1,var1)
   print "fi shape: ", fi1.shape  
   esti1_y= np.dot(fi1,weight1)
   #print "\n esti1_y shape:",esti1_y.shape   
   #print esti1_y
       
   mse1 = MSE(esti1_y,train1_y)
   print "Mean squared error of train1_set:",mse1 
   
   fi1_test=rbf_kMeans_Fi(test_x,k1,mean1,var1)
   esti_fin=np.dot(fi1_test,weight1)
   
   mes1_test=MSE(esti_fin,test_y)
   print "Mean squared error of test set : ",mes1_test                              
   
                                       
                                                                           
                                                                                                                                                                                     
   ####cis_train2 
   k2=40
   mean2, var2 = kMeans(train2_x,k2)  
   weight2=rbf_kMeans_W(train2_x,train2_y,k2,mean2,var2)
   fi2=rbf_kMeans_Fi(train2_x,k2,mean2,var2)
   print "fi shape: ", fi2.shape  
   esti2_y= np.dot(fi2,weight2)
   #print "\n esti2_y shape:",esti2_y.shape
    
   mse2 = MSE(esti2_y,train2_y)
   print "Mean squared error of train2_set:",mse2 
   
   fi2_test=rbf_kMeans_Fi(test_x,k2,mean2,var2)
   esti_fin=np.dot(fi2_test,weight2)
   
   mes2_test=MSE(esti_fin,test_y)
   print "Mean squared error of test set : ",mes2_test  