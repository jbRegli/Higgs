"""
Gathered the function for the pre-tratment
"""
import numpy as np


def split_8(ID, x, y= None, weights= None):
    """
    Given a dataset split it in 8 sub-datasets (according to some parameter defined or
    not
    """
    ID_s = []
    xs_s = []
    y_s = []
    weights_s =[]

    for n in range(8):
        ID_s.append(np.zeros(0))
        xs_s.append(np.zeros((0,x.shape[1])))
        y_s.append(np.zeros(0))
        weights_s.append(np.zeros(0))

    # Splitting the set:
    for i in range(x.shape[0]):
        if x[i,0] != -999:
            if x[i,22] == 0:
                ID_s[0]      = np.append(ID_s[0], ID[i])
                xs_s[0]      = np.vstack([xs_s[0], x[i,:]])
                if y != None:
                    y_s[0]       = np.append(y_s[0], y[i])
                if weights != None:
                    weights_s[0] = np.append(weights_s[0], weights[i])

            elif x[i,22] == 1:
                ID_s[1]      = np.append(ID_s[1], ID[i])
                xs_s[1]      = np.vstack([xs_s[1], x[i,:]])
                if y != None:
                    y_s[1]   = np.append(y_s[1], y[i])
                if weights != None:
                    weights_s[1] = np.append(weights_s[1], weights[i])

            elif x[i,22] == 2:
                ID_s[2]      = np.append(ID_s[2], ID[i])
                xs_s[2]      = np.vstack([xs_s[2], x[i,:]])
                if y != None:
                    y_s[2]   = np.append(y_s[2], y[i])
                if weights != None:
                    weights_s[2] = np.append(weights_s[2], weights[i])

            elif x[i,22] == 3:
                ID_s[3]      = np.append(ID_s[3], ID[i])
                xs_s[3]      = np.vstack([xs_s[3], x[i,:]])
                if y != None:
                    y_s[3]   = np.append(y_s[3], y[i])
                if weights != None:
                    weights_s[3] = np.append(weights_s[3], weights[i])
            else:
                print("Error: Unexpected value for column 22...")
                exit()

        else:
            if x[i,22] == 0:
                ID_s[4]      = np.append(ID_s[4], ID[i])
                xs_s[4]      = np.vstack([xs_s[4], x[i,:]])
                if y != None:
                    y_s[4]   = np.append(y_s[4], y[i])
                if weights != None:
                    weights_s[4] = np.append(weights_s[4], weights[i])

            elif x[i,22] == 1:
                ID_s[5]      = np.append(ID_s[5], ID[i])
                xs_s[5]      = np.vstack([xs_s[5], x[i,:]])
                if y != None:
                    y_s[5]   = np.append(y_s[5], y[i])
                if weights != None:
                    weights_s[5] = np.append(weights_s[5], weights[i])

            elif x[i,22] == 2:
                ID_s[6]      = np.append(ID_s[6], ID[i])
                xs_s[6]      = np.vstack([xs_s[6], x[i,:]])
                if y != None:
                    y_s[6]   = np.append(y_s[6], y[i])
                if weights != None:
                    weights_s[6] = np.append(weights_s[6], weights[i])

            elif x[i,22] == 3:
                ID_s[7]      = np.append(ID_s[7], ID[i])
                xs_s[7]      = np.vstack([xs_s[7], x[i,:]])
                if y != None:
                    y_s[7]   = np.append(y_s[7], y[i])
                if weights != None:
                    weights_s[7] = np.append(weights_s[7], weights[i])
            else:
                print("Error: Unexpected value for column 22...")
                exit()

    if y != None:
        if weights != None:
            return ID_s, xs_s, y_s, weights_s
        else:
            print ("Not a normal splitting case...")
            exit()
    else:
        if weights == None:
            return ID_s, xs_s
        else:
            print ("Not a normal splitting case...")
            exit()


def normalize(x_train, x_test):
    """
    Given a train set and a test set, normalize them without taking into account the
    undefined variables (ie x[i][j] = -999)
    The test set is normalized using the smean and the variance of the train set
    """

    # Memorize the shape of the input t test if the output's shape has not been
    # modified
    shape = x_train.shape

    # Transpose the input to work on line
    x_train = x_train.T
    x_test = x_test.T

    for i in xrange(x_train.shape[0]):
        # Don't normalize the column 0: envent id

        # Don't normalize the column 22: Decision parameter
        if i != 22 & i!= 0:
            # Normalize the data without taking into account the -999:
            mean = np.mean(x_train[i][x_train[i]!=-999.])
            variance = np.var(x_train[i][x_train[i]!=-999.])

            x_train[i][x_train[i]!=-999.] -= mean
            x_train[i][x_train[i]!=-999.] /= variance

            x_test[i][x_test[i]!=-999.] -= mean
            x_test[i][x_test[i]!=-999.] /= variance

    # Transpose back to return the same shape
    x_train= x_train.T
    x_test = x_test.T

    # Test:
    if x_train.shape != shape:
        print("Error in the normalization  x.input and x.output have different shapes")
        exit()

    return x_train, x_test


def add_noise(x):
    """
    Given a dataset, add noise to it
    """
    x = np.add(x, np.random.normal(0.0, noise_variance, x.shape))


def split_8_matrix(ID, x, y= None, weights= None):
    """
    Given a dataset split it in 8 sub-datasets (according to some parameter defined or
    not
    """
    ID_s = []
    xs_s = []

    if y != None:
        y_s = []
    if weights != None:
        weights_s =[]

    for n in range(8):
        ID_s.append(np.zeros(0))
        xs_s.append(np.zeros((0,x.shape[1])))
        if y != None:
            y_s.append(np.zeros(0))
        if weights != None:
            weights_s.append(np.zeros(0))

    # Splitting the set:
    # Criteria 1: Is the first column equal to -999?
    # Criteria 2: Is the 22th column equal to 0, 1, 2 or 3?

    # ID_s:
    ID_s[0] = ID[np.logical_and(x[:,0]!=-999., x[:,22]==0.)]
    ID_s[1] = ID[np.logical_and(x[:,0]!=-999., x[:,22]==1.)]
    ID_s[2] = ID[np.logical_and(x[:,0]!=-999., x[:,22]==2.)]
    ID_s[3] = ID[np.logical_and(x[:,0]!=-999., x[:,22]==3.)]

    ID_s[4] = ID[np.logical_and(x[:,0]==-999., x[:,22]==0.)]
    ID_s[5] = ID[np.logical_and(x[:,0]==-999., x[:,22]==1.)]
    ID_s[6] = ID[np.logical_and(x[:,0]==-999., x[:,22]==2.)]
    ID_s[7] = ID[np.logical_and(x[:,0]==-999., x[:,22]==3.)]

    # xs_s
    xs_s[0] = x[np.logical_and(x[:,0]!=-999, x[:,22]==0)]
    xs_s[1] = x[np.logical_and(x[:,0]!=-999, x[:,22]==1)]
    xs_s[2] = x[np.logical_and(x[:,0]!=-999, x[:,22]==2)]
    xs_s[3] = x[np.logical_and(x[:,0]!=-999, x[:,22]==3)]

    xs_s[4] = x[np.logical_and(x[:,0]==-999, x[:,22]==0)]
    xs_s[5] = x[np.logical_and(x[:,0]==-999, x[:,22]==1)]
    xs_s[6] = x[np.logical_and(x[:,0]==-999, x[:,22]==2)]
    xs_s[7] = x[np.logical_and(x[:,0]==-999, x[:,22]==3)]

    # y_s
    if y != None:
        y_s[0] = y[np.logical_and(x[:,0]!=-999, x[:,22]==0)]
        y_s[1] = y[np.logical_and(x[:,0]!=-999, x[:,22]==1)]
        y_s[2] = y[np.logical_and(x[:,0]!=-999, x[:,22]==2)]
        y_s[3] = y[np.logical_and(x[:,0]!=-999, x[:,22]==3)]

        y_s[4] = y[np.logical_and(x[:,0]==-999, x[:,22]==0)]
        y_s[5] = y[np.logical_and(x[:,0]==-999, x[:,22]==1)]
        y_s[6] = y[np.logical_and(x[:,0]==-999, x[:,22]==2)]
        y_s[7] = y[np.logical_and(x[:,0]==-999, x[:,22]==3)]

    if weights != None:
        weights_s[0] = weights[np.logical_and(x[:,0]!=-999, x[:,22]==0)]
        weights_s[1] = weights[np.logical_and(x[:,0]!=-999, x[:,22]==1)]
        weights_s[2] = weights[np.logical_and(x[:,0]!=-999, x[:,22]==2)]
        weights_s[3] = weights[np.logical_and(x[:,0]!=-999, x[:,22]==3)]

        weights_s[4] = weights[np.logical_and(x[:,0]==-999, x[:,22]==0)]
        weights_s[5] = weights[np.logical_and(x[:,0]==-999, x[:,22]==1)]
        weights_s[6] = weights[np.logical_and(x[:,0]==-999, x[:,22]==2)]
        weights_s[7] = weights[np.logical_and(x[:,0]==-999, x[:,22]==3)]


    if y != None:
        if weights != None:
            return ID_s, xs_s, y_s, weights_s
        else:
            print ("Not a normal splitting case...")
            exit()
    else:
        if weights == None:
            return ID_s, xs_s
        else:
            print ("Not a normal splitting case...")
            exit()

def concatenate_vectors(vector_s):
    """
    concatenate the vectors in vector_s and returns the concatenated vector
    """
    if len(vector_s[0].shape)>1:
        concatenated_vector = np.empty((0, vector_s[0].shape[1])) # empty vector with the same number of columns than the arrays in the list
    else:
        concatenated_vector = np.empty(0)
    for vector in vector_s:
        concatenated_vector = np.concatenate((concatenated_vector, vector))

    return concatenated_vector

def ratio_sig_per_dataset(y_s):
    """
    Compute the percentage of signal among a given dataset
    """
    # If we work with the splitted dataset:
    if type(y_s) == list:
        average = []
        for i in range(len(y_s)):
            average.append(float(np.sum(y_s[i]))/(y_s[i].shape[0]))
            print ("dataset %i: %i elements - %.2f%% of signal.") \
                    %(i, y_s[i].shape[0], average[-1]*100)
    else:
        average = float(np.sum(y_s))/(y_s.shape[0])
        print ("dataset: %i elements - %.2f%% of signal.") \
                    %(y_s.shape[0], average)

    return average

def binary2multiclass(yBinary, weights):
    """
    function that gives a multiclass label vector
    yBinary = vector of the binary labels
    weights = vector of the weights
    returns a vector with 5 classes : {0,1,2,3} = different events signal, 4 = non event signals
    """
    if yBinary.shape != weights.shape:
        print "Vector of labels and Vector of weights must be the same size"
        exit()

    yMultiClass = np.zeros_like(yBinary)

    for i, (y, weight) in enumerate(zip(yBinary, weights)):
        if y ==1:
            if weight ==0.018636116671999998:
                yMultiClass[i] = 1
            elif weight ==0.0015027048310100001:
                yMultiClass[i] = 2
            elif weight ==0.0026533113373300001:
                yMultiClass[i] = 3
            elif weight ==0.00150187015894:
                yMultiClass[i] = 4
        else:
            yMultiClass[i] = 0

    return yMultiClass

def multiclass2binary(yMulticlass):
    """
    function that transforms a multiclass label vectors into a binary one
    """
    yBinary = np.zeros(yMulticlass.shape[0])
    for i in range(yMulticlass.shape[0]):
        if yMulticlass[i] >=1:
            yBinary[i] = 1
    
    return yBinary



