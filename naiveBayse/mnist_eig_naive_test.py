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
    
def mean_var(train_x):
    mean_var = [(np.mean(attr),np.var(attr)) for attr in zip(*train_x)]
    del mean_var[-1]
    return mean_var
       
   
def classValue(train_x):
    class_mean_var={}
    for index, instance in train_x.iteritems():
        class_mean_var[index]=mean_var(instance)
    return class_mean_var
    
def accu_knn(classify_x,test_y) :
     """return the accuracy of knn"""
     accu=float(sum(classify_x[i]==test_y[i] for i in range(len(classify_x))))/len(classify_x)
     #print sum(classify_x[i]==train_y[i] for i in range(len(classify_x)))
     return accu
     
def gausian_prob(x,mean,var):
    """retrun value of gausian distribution"""  
    exponent = math.exp(-(math.pow(float(x)-float(mean),2)/(2*var)))
    gausian_P = (1/(math.sqrt(2*math.pi*var)))*exponent
    return gausian_P
    

if __name__ == '__main__':
    train_set, val_set, test_set = load_data('mnist.pkl.gz')

    train_x, train_y = train_set
    val_x, val_y = val_set
    test_x, test_y = test_set
    
    N, D = train_x.shape
    print train_x.shape
    print train_y.shape
    test_x_1000=test_x[:100,:]
    
    index_set = setIndex(train_x,train_y)
    print "index_set : ",len(index_set)
    
    mean_list=[]
    var_list=[]
    
    
    for col in range(train_x.shape[1]):
        mean_list.append([])
        var_list.append([])
        for j in range(10):
            class_value = classValue(train_x,index_set[j],col)
            mean_list[col].append(np.mean(class_value))
            var_list[col].append(np.var(class_value))
    mean_arr=np.asarray(mean_list)
    var_arr=np.asarray(var_list)
    print "mean_arr.shape : ",mean_arr.shape
    print "var_arr.shape : ",var_arr.shape
    
    for i in range(var_arr.shape[0]):
        for j in range(10):
            if var_arr[i,j] == 0:
                var_arr[i,j]=1               #when variance==0 convert =0.00001(very small number)
                
    """ Naive Baysian classificaion for test_x"""

    
    """train_eigen space"""
    #train_x_zero = train_x - np.matlib.repmat(train_x.mean(0), N, 1)
    #cov_mat = np.dot(train_x_zero.transpose(), train_x_zero)/N

    # for eigendecomposition 
    # http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.eig.html 
    #w, v = LA.eig(cov_mat)
    #print v.shape
    #print w.shape
    
    accu_nai = accu_knn(test_nai_clsi,test_y)
    print "accu: ", accu_nai

