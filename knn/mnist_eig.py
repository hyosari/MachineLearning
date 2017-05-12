import six.moves.cPickle as pickle
import gzip
import os
import numpy as np
import scipy.misc
from numpy import linalg as LA
import numpy.matlib
import matplotlib.pyplot as plt


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


if __name__ == '__main__':
    train_set, val_set, test_set = load_data('mnist.pkl.gz')

    train_x, train_y = train_set
    val_x, val_y = val_set
    test_x, test_y = test_set
    
    N, D = train_x.shape
    print train_x.shape
    print train_y.shape
    print train_y[0]

    mean_img = train_x.mean(0).reshape((28, 28))
    scipy.misc.imsave('mean.jpg', mean_img)
    var_img = train_x.var(0).reshape((28, 28))
    scipy.misc.imsave('var.jpg', var_img)

    train_x_zero = train_x - np.matlib.repmat(train_x.mean(0), N, 1)
    cov_mat = np.dot(train_x_zero.transpose(), train_x_zero)/N

    # for eigendecomposition 
    # http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.eig.html 
    w, v = LA.eig(cov_mat)
    print v.shape
    print w.shape
    for i in range(10):
        e_img = v[:,i].reshape((28, 28))
        scipy.misc.imsave('eig_v_large' + str(i) + '.jpg', e_img)
        e_img = v[:,783-i].reshape((28, 28))
        scipy.misc.imsave('eig_v_small' + str(i) + '.jpg', e_img)
    
    plt.plot(np.arange(D), w)
    plt.savefig('eig_val.png')

