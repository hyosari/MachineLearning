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
        for j in range(train_x.shape[1]):
            class_k_mean[i].append(np.mean(train_x[class_k[i],j]))  
     class_k_mean=np.array(class_k_mean) # mean nparray

  
     if (points == class_k_mean).all() == False:
        points = class_k_mean
        continue    
     else:
         break
    #class_k_var =np.repeat([0.001],k)
    class_k_var =estiVar(class_k_mean,class_k,train_x)     
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
    return var_temp*var_temp     

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
        
    
    weight=np.dot(np.dot(inv(np.dot(np.transpose(fi),fi)),np.transpose(fi)),train_y)
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
                              
if __name__ == '__main__':
#load text data
   cis_train1= loadtxt("cis_train1.txt",delimiter="\t",unpack=False)
   cis_train2= loadtxt("cis_train2.txt",delimiter="\t",unpack=False)
   cis_test= loadtxt("cis_test.txt",delimiter="\t",unpack=False)

   print "cis_train1: ",cis_train1.shape
   print "cis_train2: ",cis_train2.shape
   print "cis_test: ",cis_test.shape


   train1_x,train1_y = cis_train1[:,:2],cis_train1[:,2]
   train2_x,train2_y = cis_train2[:,:2],cis_train2[:,2]
   test_x,test_y = cis_test[:,:2],cis_test[:,2]
  
   ####cis_train1
   k1=15
   mean1, var1 = kMeans(train1_x,k1)
   weight1=rbf_kMeans_W(train1_x,train1_y,k1,mean1,var1)
   fi1=rbf_kMeans_Fi(train1_x,k1,mean1,var1)
   print "fi shape: ", fi1.shape  
   esti1_y= np.dot(fi1,weight1)
   #print "\n esti1_y shape:",esti1_y.shape   
   esti1=[]
   for i in range(train1_x.shape[0]):
       if esti1_y[i]>0.5:
           esti1.append(1)
       else:
            esti1.append(0)
       
   accu1=0.0
   for i in range(train1_x.shape[0]): 
       if esti1[i]==train1_y[i]:
           accu1+=1.0        
   print "\ntrain1_accuracy:",accu1/train1_x.shape[0]
               
   fi_t=rbf_kMeans_Fi(test_x,k1,mean1,var1)
   esti_final = np.dot(fi_t,weight1)                
   accu_fin=0.0                 
   for i in range(test_x.shape[0]):
       if esti_final[i] >0.5:
           if test_y[i]==1:
               accu_fin+=1
       else:
           if test_y[i]==0:
               accu_fin+=1                        
                               
   print "test set with train1 weight accuracy:",accu_fin/test_x.shape[0]                                
                                       
                                               
   ####cis_train2 
   k2=55 
   mean2, var2 = kMeans(train2_x,k2)  
   weight2=rbf_kMeans_W(train2_x,train2_y,k2,mean2,var2)
   fi2=rbf_kMeans_Fi(train2_x,k2,mean2,var2)
   print "fi shape: ", fi2.shape  
   esti2_y= np.dot(fi2,weight2)
   #print "\n esti2_y shape:",esti2_y.shape
    
   esti2=[]
   for i in range(train2_x.shape[0]):
       if esti2_y[i]>0.5:
           esti2.append(1)
       else:
            esti2.append(0)
   accu2=0.0
   for i in range(train2_x.shape[0]): 
       if esti2[i]==train2_y[i]:
           accu2+=1.0
   print "\ntrain2 accuracy:",accu2/train2_x.shape[0]
      
   fi_t=rbf_kMeans_Fi(test_x,k2,mean2,var2)
   esti_final = np.dot(fi_t,weight2)
   
   accu_fin=0.0
   for i in range(test_x.shape[0]):
       if esti_final[i] >0.5:
           if test_y[i]==1:
               accu_fin+=1
       else:
           if test_y[i]==0:
               accu_fin+=1

   print "test set with train2 weight accuracy:",accu_fin/test_x.shape[0] 
   

   

            
  
      