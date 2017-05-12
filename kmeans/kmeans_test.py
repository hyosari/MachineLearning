# -*- coding: utf-8 -*-
import six.moves.cPickle as pickle
import gzip
import os
import numpy as np
import scipy.misc
from numpy import linalg as LA
import numpy.matlib
import matplotlib.pyplot as plt
import pylab as pl
import scipy.spatial.distance as dis
import random
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
    
def cal_dist(x,y):
    dis=0.0
    if len(x) != len(y):
        print "Not same Col number"
        return
    for i in range(len(x)):
        dis =dis+ pow(x[i]-y[i],2)

    return math.sqrt(dis) 
            
def initial_point(ranPoint,k,train_y_39,train_x):
     dis_sort=[]
     seconPoint = np.argmax(dis_sort)# second point is that the farest point from first point 
     class_2Point = train_x[seconPoint]
     for i in range(len(train_y_39)):
        dis_sort.append(cal_dist(train_x[ranPoint],train_x[train_y_39[i]]))
     if k == 2:
        return class_2Point
     elif k==3:
        class_3Point=[]
        for i in range(len(train_x[1])):
            class_3Point.append((train_x[ranPoint,i]+class_2Point[i])/2)
        return class_2Point, np.array(class_3Point) 
     elif k==5:
        a,b,c,d = random.choice(train_y_39,4)
        return train_x[a],train_x[b],train_x[c],train_x[d]
     elif k==10:
        a,b,c,d,e,f,g,h,i = random.choice(train_y_39,9)
        return train_x[a],train_x[b],train_x[c],train_x[d],train_x[e],train_x[f],train_x[g],train_x[h],train_x[i]
     else:
         print "error k " 
         return 0
        
        
          


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
    
    train_x_origin=train_x
    #train_x = np.dot(train_x,v[:,:2])  #eigen space_2 
    #train_x = np.dot(train_x,v[:,:5])   #eigen space_5
    train_x=np.dot(train_x,v[:,:10])     #eigen space_10
    
    train_y_39=[]
    for i in range(len(train_y)):
        if train_y[i]==3 or train_y[i]==9:
            train_y_39.append(i)   # len(train_y_39) = 10089
            
    ranPoint = random.choice(train_y_39)   # first point index Random
    class_1Point = train_x[ranPoint]
    
    
    dis_sort=[]
    for i in range(len(train_y_39)):
        dis_sort.append(cal_dist(train_x[ranPoint],train_x[train_y_39[i]]))
    
    seconPoint = np.argmax(dis_sort)# second point is that the farest point from first point 
    class_2Point = train_x[seconPoint]
    lop = True 
    
    while lop : 
        class_1=[]
        class_2=[]
        for i in range(len(train_y_39)):
            dis_1p = cal_dist(class_1Point,train_x[train_y_39[i]])
            dis_2p = cal_dist(class_2Point,train_x[train_y_39[i]])
        
            class_1.append(train_y_39[i]) if dis_1p>dis_2p else class_2.append(train_y_39[i])    ## clssification
        
        mean_1=[]
        mean_2=[]
        for i in range(train_x.shape[1]):
            mean_1.append(np.mean(train_x[class_1,i]))
            mean_2.append(np.mean(train_x[class_2,i]))
        mean_1=np.array(mean_1)
        mean_2=np.array(mean_2)
        if (mean_1==class_1Point).all() == "False" or (mean_2==class_2Point).all()=="False":
            class_1Point = mean_1
            class_2Point = mean_2
            print "change"
        else:
            print "converge"
            lop=False
        
                                           
        dim2= np.dot(train_x_origin,v[:,:2])
        #dim2= train_x
        X= dim2[train_y_39,0]
        Y= dim2[train_y_39,1]
        
        col=[]
        for i in range(len(train_y_39)):
            if train_y_39[i] in class_1:
                col.append("r")
            else:
                col.append("b")
        mark=[]
        for i in range(len(train_y_39)):
            if train_y[train_y_39[i]]==3:
                mark.append(u'+')
            else:
                mark.append(u'o')
        
        
        
        
        for _s,c,_x,_y in zip(mark,col,X,Y):
            plt.scatter(_x,_y,marker=_s,c=c)
        plt.show()
            
            
        
     
    
    
    