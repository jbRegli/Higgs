"""
Gathered the function for the pre-tratment
"""
import numpy as np

def normalize(x):
    """
    Given a dataset, normilze it without taking into account the undefined variables
    (ie x[i][j] = -999)
    """

    shape = x.shape

    # Transpose the input to work on line
    x = x.T


    for i in xrange(x.shape[0]):

        # Don't normalize the column 22: Decision parameter
        if i != 22:
            # Normalize the data without taking into account the -999:
            x[i][x[i]!=-999.] -= np.mean(x[i][x[i]!=-999.])
            x[i][x[i]!=-999.] /= np.var(x[i][x[i]!=-999.])

        #print (x[i][x[i]==-999.].shape)

    # Transpose back to return the same shape
    x= x.T

    if x.shape != shape:
        print("Error in the normalization  x.input and x.output have different shapes")
        exit()

    return x


def add_noise(x):
    """
    Given a dataset, add noise to it
    """
    x = np.add(x, np.random.normal(0.0, noise_variance, x.shape))




