# -*- coding: utf-8 -*-
"""
function to visualise the AMS metrics
"""
import numpy as np
import HiggsBosonCompetition_AMSMetric_rev1 as hbc
from numpy import exp,arange
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

def ratio_event(y):
    """
    takes an vector of label (0 or 1) and returns the percentage of events
    """
    return 100*sum(y)/y.shape[0]

def print_AMS(y, weights, n_sample = 5, plot_type = "all", max_AMS = 4 ):
    """
    plot AMS = f(s,b), where s and b are the percentage of  true and fase positive
    y is the vector of label
    weights is the vectot of weights
    Both must be the same length
    plot_type : string in ["2D", "3D", "all"]
    max_AMS = max of the AMS (for the color scale)
    n_sample = number of samples to estimate s and b
    """
    plot_type_s = ["2D", "3D", "all"] #List of the admissible strings for plot_types
    if plot_type not in plot_type_s:
        print "plot type must be a string in {2D, 3D, all}"
        exit()

    sum_1 = np.sum(y)
    sum_0 = y.shape[0] - sum_1

    def get_AMS_estimation(s_p, b_p):
        """
        s_p : pourcentage d'évènements detectes
        b_p : pourcentage de faux positifs 
        """
        # We shuffle the label and weights vectors in order not to have
        # always the same examples
        s = 0.
        b = 0.

        for n in range(n_sample):
            perm = np.random.permutation(y.shape[0])
            y_shuffled = y[perm]
            weights_shuffled = weights[perm]

            y_s_tronque = np.zeros_like(y)
            compteur_s = 0
            for i in range(y.shape[0]):
                if y_shuffled[i] == 1 and compteur_s < int(s_p*sum_1):
                    compteur_s += 1
                    y_s_tronque[i] = 1
            s += np.dot(y_s_tronque, weights_shuffled)

            y_b_tronque = np.zeros_like(y)
            compteur_b = 0
            for i in range(y.shape[0]):
                if y_shuffled[i] == 0 and compteur_b < int(b_p*sum_0):
                    compteur_b += 1
                    y_b_tronque[i] = 1
            b += np.dot(y_b_tronque, weights_shuffled)

        s /= n_sample
        b /= n_sample

        s *= 250000/y.shape[0]
        b *= 250000/y.shape[0]

        return hbc.AMS(s,b)
    
    s_p = arange(0.,1.,0.05)
    b_p = arange(0.,1.0,0.05)
    X,Y = meshgrid(s_p, b_p) # grid of point
    Z = np.zeros((20,20))
    for i in range(20):
        for j in range(20):
            Z[i,j] = get_AMS_estimation(s_p[i], b_p[j])
            if Z[i,j] >  max_AMS:
                Z[i,j] = max_AMS
    
    if plot_type == "2D" or plot_type == "all":
        im = imshow(Z,cmap=cm.RdBu, origin = "lower" ) # drawing the function
        # adding the Contour lines with labels
        cset = contour(Z,arange(-1,1.5,0.2),linewidths=2,cmap=cm.Set2)
        clabel(cset,inline=True,fmt='%1.1f',fontsize=10)
        colorbar(im) # adding the colobar on the right
        # latex fashion title
        title('AMS')
    
    
    if plot_type == "3D" or plot_type == "all":
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                      cmap=cm.RdBu,linewidth=0, antialiased=False)

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()



