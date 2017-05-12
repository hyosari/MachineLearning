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

# 두점 사이의 길이를 구한다
def cal_dist(x,y):
    dis=0.0
    if len(x) != len(y):
        print "Not same Col number"
        return
    for i in range(len(x)):
        dis =dis+ pow(x[i]-y[i],2)

    return math.sqrt(dis) 

# 처음 POINT 를 설정하는 함수 
# 2,3 일때는 각 점들의 거리가 최대가 되게 한다. 
# 3이상 부터는 RANDOM PICK            
def initial_point(ranPoint,k,train_y_39,train_x):
          
     if k == 2:
        dis_sort=[]
        for i in range(len(train_y_39)):
            dis_sort.append(cal_dist(train_x[ranPoint],train_x[train_y_39[i]]))
        seconPoint = np.argmax(dis_sort)
        class_2Point = train_x[seconPoint]
        return class_2Point
     elif k==3:
        dis_sort=[]
        for i in range(len(train_y_39)):
            dis_sort.append(cal_dist(train_x[ranPoint],train_x[train_y_39[i]]))
        seconPoint = np.argmax(dis_sort)
        class_2Point = train_x[seconPoint]    
        class_3Point=[]
        for i in range(len(train_x[1])):
            class_3Point.append((train_x[ranPoint,i]+class_2Point[i])/2)
        return class_2Point, np.array(class_3Point) 
     elif k==5:
        a,b,c,d = random.sample(train_y_39,4)
        return train_x[a],train_x[b],train_x[c],train_x[d]
     elif k==10:
        a,b,c,d,e,f,g,h,i = random.sample(train_y_39,9)
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
            
    ranPoint = random.choice(train_y_39)   #  randomly initial point for class1
    class_1Point = train_x[ranPoint]
    
    
    #dis_sort=[]
    #for i in range(len(train_y_39)):
    #   dis_sort.append(cal_dist(train_x[ranPoint],train_x[train_y_39[i]]))
    
    #seconPoint = np.argmax(dis_sort)# second point is that the farest point from first point 
    #class_2Point = train_x[seconPoint]
    
    class_2Point,class_3Point,class_4Point,class_5Point,class_6Point,class_7Point,class_8Point,class_9Point,class_10Point=initial_point(ranPoint,10,train_y_39,train_x)
    
    
    
    lop = True 
    
    while lop : 
        class_1=[]
        class_2=[]
        class_3=[]
        class_4=[]
        class_5=[]
        class_6=[]
        class_7=[]
        class_8=[]
        class_9=[]
        class_10=[]
        for i in range(len(train_y_39)):
            dis_1p = cal_dist(class_1Point,train_x[train_y_39[i]])
            dis_2p = cal_dist(class_2Point,train_x[train_y_39[i]])
            dis_3p = cal_dist(class_3Point,train_x[train_y_39[i]])
            dis_4p = cal_dist(class_4Point,train_x[train_y_39[i]])
            dis_5p = cal_dist(class_5Point,train_x[train_y_39[i]])
            dis_6p = cal_dist(class_6Point,train_x[train_y_39[i]])
            dis_7p = cal_dist(class_7Point,train_x[train_y_39[i]])
            dis_8p = cal_dist(class_8Point,train_x[train_y_39[i]])
            dis_9p = cal_dist(class_9Point,train_x[train_y_39[i]])
            dis_10p = cal_dist(class_10Point,train_x[train_y_39[i]])
            
            dis_min=[dis_1p,dis_2p,dis_3p,dis_4p,dis_5p,dis_6p,dis_7p,dis_8p,dis_9p,dis_10p]
            max_cls=np.argmin(dis_min)
        # class_1.append(train_y_39[i]) if dis_1p>dis_2p else class_2.append(train_y_39[i])     ## clssification
            if max_cls==0 :
               class_1.append(train_y_39[i]) 
            elif max_cls==1 :
                class_2.append(train_y_39[i])  
            elif  max_cls==2 :
                class_3.append(train_y_39[i]) 
            elif max_cls==3 :
                class_4.append(train_y_39[i])
            elif max_cls==4 :
                class_5.append(train_y_39[i]) 
            elif max_cls==5 :
                class_6.append(train_y_39[i])
            elif max_cls==6 :
                class_7.append(train_y_39[i])
            elif max_cls==7 :
                class_8.append(train_y_39[i])
            elif max_cls==8 :
                class_9.append(train_y_39[i])
            else :
                class_10.append(train_y_39[i]) 
        
        
        
        mean_1=[]
        mean_2=[]
        mean_3=[]
        mean_4=[]
        mean_5=[]
        mean_6=[]
        mean_7=[]
        mean_8=[]
        mean_9=[]
        mean_10=[]
        for i in range(train_x.shape[1]):
            mean_1.append(np.mean(train_x[class_1,i]))
            mean_2.append(np.mean(train_x[class_2,i]))
            mean_3.append(np.mean(train_x[class_3,i]))
            mean_4.append(np.mean(train_x[class_4,i]))
            mean_5.append(np.mean(train_x[class_5,i]))
            mean_6.append(np.mean(train_x[class_6,i]))
            mean_7.append(np.mean(train_x[class_7,i]))
            mean_8.append(np.mean(train_x[class_8,i]))
            mean_9.append(np.mean(train_x[class_9,i]))
            mean_10.append(np.mean(train_x[class_10,i]))
        mean_1=np.array(mean_1)
        mean_2=np.array(mean_2)
        mean_3=np.array(mean_3)
        mean_4=np.array(mean_4)
        mean_5=np.array(mean_5)
        mean_6=np.array(mean_6)
        mean_7=np.array(mean_7)
        mean_8=np.array(mean_8)
        mean_9=np.array(mean_9)
        mean_10=np.array(mean_10)
        if (mean_1==class_1Point).all() == "False" or (mean_2==class_2Point).all()=="False" or (mean_3==class_3Point).all()=="False" or (mean_4==class_4Point).all()=="False" or (mean_5==class_5Point).all()=="False" or (mean_6==class_6Point).all()=="False"or (mean_7==class_7Point).all()=="False"or (mean_8==class_8Point).all()=="False"or (mean_9==class_9Point).all()=="False"or (mean_10==class_10Point).all()=="False":
            class_1Point = mean_1
            class_2Point = mean_2
            class_3Point = mean_3
            class_4Point = mean_4
            class_5Point = mean_5
            class_6Point = mean_6
            class_7Point = mean_7
            class_8Point = mean_8
            class_9Point = mean_9
            class_10Point = mean_10
            
            print "change"
        else:
            print "converge"
            lop=False
        
                                           
        dim2= np.dot(train_x_origin,v[:,:2])
        #dim2=train_x
        X= dim2[train_y_39,0]
        Y= dim2[train_y_39,1]
        
        col=[]
        for i in range(len(train_y_39)):
            if train_y_39[i] in class_1:
                col.append("black")
            elif train_y_39[i] in class_2:
                col.append("r")
            elif train_y_39[i] in class_3 :
                col.append("sandybrown")
            elif train_y_39[i] in class_4:
                col.append("gold")
            elif train_y_39[i] in class_5:
                col.append("chartreuse")
            elif train_y_39[i] in class_6:
                col.append("green")
            elif train_y_39[i] in class_7:
                col.append("darkslategrey") 
            elif train_y_39[i] in class_8:
                col.append("c") 
            elif train_y_39[i] in class_9:
                col.append("dodgerblue") 
            else :
                col.append("purple") 
        mark=[]
        for i in range(len(train_y_39)):
            if train_y[train_y_39[i]]==3:
                mark.append(u'+')
            else:
                mark.append(u'o')
        
        
        
        
        for _s,c,_x,_y in zip(mark,col,X,Y):
            plt.scatter(_x,_y,marker=_s,c=c)
        plt.show()
            