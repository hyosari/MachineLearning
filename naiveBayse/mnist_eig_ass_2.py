# -*- coding: utf-8 -*-
import six.moves.cPickle as pickle
import gzip
import os
import numpy as np
import scipy.misc
from scipy.stats import norm 
from numpy import linalg as LA
import numpy.matlib
import matplotlib.pyplot as plt
import math


def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    
    copied from http://deeplearning.net/ and revised by hchoi
    '''

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        from six.moves import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    print('... loading data')

    # Load the dataset
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.

    return train_set, valid_set, test_set
    
    
    
def setIndex(train_x,train_y):
    """gather index about each class"""
    index_set=[]
    for i in range(10):
        index_set.append([])

    for i in range(10):
        for j in range(train_x.shape[0]):
            if train_y[j] == i :
                index_set[i].append(j)
     
    return index_set
    
def classValue(train_x,index_set,col):
    """index set's colunm value """
    class_value=[]
    for k in range(len(index_set)):
        class_value.append(train_x[index_set[k],col])
    return class_value
    
def cal_mean_var(train_x,index_set):
    mean_list=[]
    var_list=[]
    for col in range(train_x.shape[1]):
        mean_list.append([])
        var_list.append([])
        for j in range(10):
            class_value = classValue(train_x,index_set[j],col)
            mean_list[col].append(np.mean(class_value))
            var_list[col].append(np.var(class_value))
            #print "mean value of ",j,"is ",np.mean(class_value)
            #print "val value of ",j, "is ",np.var(class_value)
            #print "generate mean and var list [",col,"]"
    mean_arr=np.asarray(mean_list)
    var_arr=np.asarray(var_list)
    #print "mean_arr shape : ",mean_arr.shape
    #print "mean_arr ", mean_arr
    #print "mean_list ", mean_list
    return mean_arr,var_arr
    
def accu_bayse(classify_x,test_y) :
     """return the accuracy of knn"""
     accu=float(sum(classify_x[i]==test_y[i] for i in range(len(classify_x))))/len(classify_x)
     #print sum(classify_x[i]==train_y[i] for i in range(len(classify_x)))
     return accu
     
def gausian_prob(x,mean,var):
    """retrun value of gausian distribution"""  
    exponent = math.exp(-(math.pow(float(x)-float(mean),2)/(2*var)))
    gausian_P = (1/(math.sqrt(2*math.pi*var)))*exponent
    return gausian_P
    
def nayveBay(test_x_1000,mean_arr,var_arr):
    #naiveBayies
     bay_esti=[]
     for j in range(test_x_1000.shape[0]):
         class_pro=[]
         for i in range(10):
             probability=1
             for k in range(test_x_1000.shape[1]):
                 if (var_arr[k,i]==0):
                     temp=1
                 else:
                   temp = gausian_prob(test_x_1000[j,k],mean_arr[k,i],var_arr[k,i] )
                   if(temp>1):temp=1
                #print "about",test_x_1000[j,k]," ",i,"class",k,"th gaussian property value : ",temp
                 if(float(temp)==0): break;   
                 probability *= temp   
           # print j," prod ",probability,"row and class ",i
             class_pro.append(probability)
         bay_esti.append(np.argmax(class_pro))
       # print "argmax(=fianl class of ",j,": ",np.argmax(class_pro)
         print "loop",j
     test_nai_clsi = np.asarray(bay_esti)
     return test_nai_clsi
    

if __name__ == '__main__':
    train_set, val_set, test_set = load_data('mnist.pkl.gz')

    train_x, train_y = train_set
    val_x, val_y = val_set
    test_x, test_y = test_set
    
    N, D = train_x.shape
    print train_x.shape
    print train_y.shape
    test_x_1000=test_x[:1000,:]
    
    index_set = setIndex(train_x,train_y)
    print "index_set : ",len(index_set)

    
                
    """ Naive Baysian classificaion for test_x"""
    #bay_esti=[]
    #for j in range(test_x_1000.shape[0]):
    #    class_pro=[]
    #    for i in range(10):
    #        probability=1
    #        for k in range(test_x_1000.shape[1]):
    #            if (var_arr[k,i]==0):
    #                temp=1
    #            else:
    #              temp = gausian_prob(test_x_1000[j,k],mean_arr[k,i],var_arr[k,i] )
    #              if(temp>1):temp=1
                #print "about",test_x_1000[j,k]," ",i,"class",k,"th gaussian property value : ",temp
     #           probability *= temp
     #           if(float(temp)==0): break;        
           # print j," prod ",probability,"row and class ",i
     #       class_pro.append(probability)
     #   bay_esti.append(np.argmax(class_pro))
       # print "argmax(=fianl class of ",j,": ",np.argmax(class_pro)
      #  print "loop",j
    #test_nai_clsi = np.asarray(bay_esti)
    #print "bay_esti : ", test_nai_clsi.shape
    
    """train_eigen space"""
    train_x_zero = train_x - np.matlib.repmat(train_x.mean(0), N, 1)
    cov_mat = np.dot(train_x_zero.transpose(), train_x_zero)/N

    # for eigendecomposition 
    # http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.eig.html 
    w, v = LA.eig(cov_mat)
    print v.shape
    print w.shape
    

    
    """train_data eigen space_2dim"""
    eigen_train_2dim = np.dot(train_x,v[:,:2])                    #train data eigen space 할때는 5000-> 50000 으로 바꿈 
    print "train_data_eigen projection  : ", eigen_train_2dim.shape          
    """2dim_test_egenspace"""
    eigen_2dim_test=np.dot(test_x_1000,v[:,:2])
    
    """5dim_train_egenspace"""
    eigen_train_5dim= np.dot(train_x,v[:,:5])  
    """5dim_test_egenspace"""
    eigen_5dim_test=np.dot(test_x_1000,v[:,:5])
    
    """10dim_train_egenspace"""
    eigen_train_10dim= np.dot(train_x,v[:,:10])  
    """10dim_test_egenspace"""
    eigen_10dim_test=np.dot(test_x_1000,v[:,:10])
    
    """raw_data """
    mean_arr,var_arr = cal_mean_var(train_x,index_set)
    print "mean_arr.shape : ",mean_arr.shape
    print "var_arr.shape : ",var_arr.shape
    
    test_nai_clsi=nayveBay(test_x_1000,mean_arr,var_arr)
    accu_nai = accu_bayse(test_nai_clsi,test_y)
    
    
    """2dim_eigen"""
    mean_arr_2dim,var_arr_2dim= cal_mean_var(eigen_train_2dim,index_set)
    test_nai_clsi_2dim=nayveBay(eigen_2dim_test,mean_arr_2dim,var_arr_2dim)
    accu_nai_2dim=accu_bayse(test_nai_clsi_2dim,test_y)
    
    
    """5dim_eigen"""
    mean_arr_5dim,var_arr_5dim= cal_mean_var(eigen_train_5dim,index_set)
    test_nai_clsi_5dim=nayveBay(eigen_5dim_test,mean_arr_5dim,var_arr_5dim)
    accu_nai_5dim=accu_bayse(test_nai_clsi_5dim,test_y)
    
    """10dim eigen"""
    mean_arr_10dim,var_arr_10dim= cal_mean_var(eigen_train_10dim,index_set)
    test_nai_clsi_10dim=nayveBay(eigen_10dim_test,mean_arr_10dim,var_arr_10dim)
    accu_nai_10dim=accu_bayse(test_nai_clsi_10dim,test_y)
    
    
    print "raw_data predict dim : ",test_nai_clsi.shape
    print "raw_data accu: ", accu_nai
    print "2dim_eigen data predict dim: ",test_nai_clsi_2dim.shape
    print "2dim_eigen data accu: ",accu_nai_2dim
    print "5dim_eigen data predict dim: ",test_nai_clsi_5dim.shape
    print "5dim_eigen data accu: ",accu_nai_5dim
    print "10dim_eigen data predict dim: ",test_nai_clsi_10dim.shape
    print "10dim_eigen data accu: ",accu_nai_10dim
    
    
    
    

