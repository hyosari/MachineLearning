# -*- coding: utf-8 -*-
import six.moves.cPickle as pickle
import gzip
import os
import numpy as np
import scipy.misc
from numpy import linalg as LA
import numpy.matlib
import matplotlib.pyplot as plt
import scipy.spatial.distance as dis
from collections import Counter


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
    
#Estimate accuracy knn 
def accu_knn(classify_x,train_y) :
     """return the accuracy of knn"""
     accu=float(sum(classify_x[i]==train_y[i] for i in range(len(classify_x))))/len(classify_x)
     #print sum(classify_x[i]==train_y[i] for i in range(len(classify_x)))
     return accu

def knn_estimate(test_x,train_y,udist,k) :
    """ knn classification"""    
    classify_x=[]
    
    for j in range(test_x.shape[0]):
        knn_set=[]   # nighborhood set 
        sort_index = np.argsort(udist[j])    #Uclian distance matrix 
        for i in range(k):
            #print "udist minimum value udist[",j,"][",i,"] :",udist[j][sort_index[i]]
            knn_set.append(train_y[sort_index[i]])
            #print sort_index[i+1]," : ",udist[j][sort_index[i+1]]
            #print bool(knn_set[i] == train_y[i] for i in range(k))
        knn_counter=Counter(knn_set)
        #print knn_counter
        classi,_ = knn_counter.most_common()[0]
        #print classi,train_y[j]
        classify_x.append(classi)  
        #print "udist[",j,"][:10] : ", udist[j][:10]
        #print "udist minimum index :",sort_index[:6]   
    return classify_x


if __name__ == '__main__':
    train_set, val_set, test_set = load_data('mnist.pkl.gz')

    train_x, train_y = train_set
    val_x, val_y = val_set
    test_x, test_y = test_set
    
    N, D = train_x.shape
    print train_x.shape
    print train_y.shape
    #print train_y[0]

    train_x_zero = train_x - np.matlib.repmat(train_x.mean(0), N, 1)
    cov_mat = np.dot(train_x_zero.transpose(), train_x_zero)/N

    # for eigendecomposition 
    # http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.eig.html 
    """train_data eigen space"""
    w, v = LA.eig(cov_mat)
    print v.shape
    print w.shape
    


    
    """raw_train_data_5000"""
    train_x_5000 = train_x[:5000,]
    print "train_x_5000 shape[0]: ", train_x_5000.shape[0]
    print "test_x.shape : ", test_x.shape
    
    """train_data_5000 eigen space_2dim"""
    eigen_train_5000_2dim = np.dot(train_x_5000,v[:,:2])                    #train data eigen space 할때는 5000-> 50000 으로 바꿈 
    print "train_data_5000 eigen projection  : ", eigen_train_5000_2dim.shape       
    
    """2dim_test_egenspace"""
    eigen_2dim_test=np.dot(test_x,v[:,:2])
    
    """5dim_train_egenspace_5000"""
    eigen_5dim_train_5000= np.dot(train_x_5000,v[:,:5])  
    """5dim_test_egenspace"""
    eigen_5dim_test=np.dot(test_x,v[:,:5])
    
    """10dim_train_egenspace_5000"""
    eigen_10dim_train_5000= np.dot(train_x_5000,v[:,:10])  
    """5dim_test_egenspace"""
    eigen_10dim_test=np.dot(test_x,v[:,:10])
    
   # udist = dis.cdist(test_x,train_x_5000)
    #print udist.shape
    #print udist[1] 
   
   # sort_index = np.argsort(udist[0])
    #for i in range(1):
    #    print udist[1][i+1]
    #    print sort_index[i+1]
    #    print sort_index[i]
    #    print train_y[sort_index[i+1]]
    
    #classify_x=[]
    
    #for j in range(5000):
     #  classify_x.append(train_y[np.argmin(np.nonzero(udist[j]))])
     
    #print np.argmin(np.nonzero(udist[1]))
    #print train_y[np.argmin(np.nonzero(udist[1]))]
    #print train_y[1]
    
       
    #accu=float(sum(classify_x[i]==train_y[i] for i in range(5000)))/5000
    
   # print accu
   
    """ raw_test set KNN"""
    udist = dis.cdist(test_x,train_x_5000)    #uclian distance
    print "distance shape 10000,5000 :", udist.shape
    print "distance with test and train_5000 " ,udist[1] 
   
    classify_x_knn1 =knn_estimate(test_x,train_y,udist,1)
    accu_k1 = accu_knn(classify_x_knn1,test_y)
    
    classify_x_knn5=knn_estimate(test_x,train_y,udist,5)
    accu_k5 = accu_knn(classify_x_knn5,test_y)
    
    classify_x_knn10=knn_estimate(test_x,train_y,udist,10)
    accu_k10= accu_knn(classify_x_knn10,test_y)
    
    print "raw_test_set Accuracy"
    print "knn1 : " ,accu_k1
    print "knn5 : ", accu_k5
    print "knn10 : ",accu_k10
    
    """2-dim eigenspace KNN"""
    udist_2dim = dis.cdist(eigen_2dim_test,eigen_train_5000_2dim)
    print "cdist matrix shape: ",udist_2dim.shape
    
    classify_x_knn1_2dim =knn_estimate(eigen_2dim_test,train_y,udist_2dim,1)
    accu_k1_2dim = accu_knn(classify_x_knn1_2dim,test_y)
    
    classify_x_knn5_2dim =knn_estimate(eigen_2dim_test,train_y,udist_2dim,5)
    accu_k5_2dim = accu_knn(classify_x_knn5_2dim,test_y)
    
    classify_x_knn10_2dim =knn_estimate(eigen_2dim_test,train_y,udist_2dim,10)
    accu_k10_2dim = accu_knn(classify_x_knn10_2dim,test_y)
    
    print "2-dim eigenspace Accuracy"
    print "knn1_2dim : " ,accu_k1_2dim
    print "knn5_2dim : ", accu_k5_2dim
    print "knn10_2dim : ",accu_k10_2dim 
    
    """5-dim eigenspace KNN"""
    udist_5dim = dis.cdist(eigen_5dim_test,eigen_5dim_train_5000)
    print "cdist matrix shape: ",udist_5dim.shape
    
    classify_x_knn1_5dim =knn_estimate(eigen_5dim_test,train_y,udist_5dim,1)
    accu_k1_5dim = accu_knn(classify_x_knn1_5dim,test_y)
    
    classify_x_knn5_5dim =knn_estimate(eigen_5dim_test,train_y,udist_5dim,5)
    accu_k5_5dim = accu_knn(classify_x_knn5_5dim,test_y)
    
    classify_x_knn10_5dim =knn_estimate(eigen_5dim_test,train_y,udist_5dim,10)
    accu_k10_5dim = accu_knn(classify_x_knn10_5dim,test_y)
    
    print "5-dim eigenspace Accuracy"
    print "knn1_5dim : " ,accu_k1_5dim
    print "knn5_5dim : ", accu_k5_5dim
    print "knn10_5dim : ",accu_k10_5dim 
    
    """10-dim eigenspace KNN"""
    udist_10dim = dis.cdist(eigen_10dim_test,eigen_10dim_train_5000)
    print "cdist matrix shape: ",udist_10dim.shape
    
    classify_x_knn1_10dim =knn_estimate(eigen_10dim_test,train_y,udist_10dim,1)
    accu_k1_10dim = accu_knn(classify_x_knn1_10dim,test_y)
    
    classify_x_knn5_10dim =knn_estimate(eigen_10dim_test,train_y,udist_10dim,5)
    accu_k5_10dim = accu_knn(classify_x_knn5_10dim,test_y)
    
    classify_x_knn10_10dim =knn_estimate(eigen_10dim_test,train_y,udist_10dim,10)
    accu_k10_10dim = accu_knn(classify_x_knn10_10dim,test_y)
    
    print "10-dim eigenspace Accuracy"
    print "knn1_10dim : " ,accu_k1_10dim
    print "knn5_10dim : ", accu_k5_10dim
    print "knn10_10dim : ",accu_k10_10dim 
    
     
   
    
                

      
      



