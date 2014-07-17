# -*- coding: utf-8 -*-

import numpy as np
import HiggsBosonCompetition_AMSMetric_rev1 as hbc
from numpy import exp,arange
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show

def ratio_event(y):
    """
    takes an vector of label (0 or 1) and returns the percentage of events
    """
    return 100*sum(y)/y.shape[0]

def print_AMS(y, weights):
    """
    plot AMS = f(s,b), where s and b are the percentage of  true and fase positive
    y is the vector of label
    weights is the vectot of weights
    Both must be the same length
    """
    sum_1 = np.sum(y)
    sum_0 = y.shape[0] - sum_1

    def get_AMS_estimation(s_p, b_p):
        """
        s_p : pourcentage d'évènements detectes
        b_p : pourcentage de faux positifs 
        """
        y_s_tronque = np.zeros_like(y)
        compteur_s = 0 
        for i in range(y.shape[0]):
            if y[i] == 1 and compteur_s < int(s_p*sum_1):
                compteur_s += 1
                y_s_tronque[i] = 1
        s = np.dot(y_s_tronque, weights)

        y_b_tronque = np.zeros_like(y)
        compteur_b = 0
        for i in range(y.shape[0]):
            if y[i] == 0 and compteur_b < int(b_p*sum_0):
                compteur_b += 1
                y_b_tronque[i] = 1

        b = np.dot(y_b_tronque, weights)

        s *= 250000/y.shape[0]
        b *= 250000/y.shape[0]

        return hbc.AMS(s,b)

    test = get_AMS_estimation(0.7, 0.2)


    # the function that I'm going to plot
    #def z_func(x,y):
    #return (1-(x**2+y**3))*exp(-(x**2+y**2)/2)
 
    s_p = arange(0.,1.,0.05)
    b_p = arange(0.,1.0,0.05)
    X,Y = meshgrid(s_p, b_p) # grid of point
    Z = np.zeros((20,20))
    for i in range(20):
        for j in range(20):
            Z[i,j] = get_AMS_estimation(s_p[i], b_p[j])
            if Z[i,j] > 4.:
                Z[i,j] = 4.

#    Z = get_AMS_estimation(X, Y) # evaluation of the function on the grid

    im = imshow(Z,cmap=cm.RdBu, origin = "lower" ) # drawing the function
    # adding the Contour lines with labels
    cset = contour(Z,arange(-1,1.5,0.2),linewidths=2,cmap=cm.Set2)
    clabel(cset,inline=True,fmt='%1.1f',fontsize=10)
    colorbar(im) # adding the colobar on the right
    # latex fashion title
    title('Prout')
    show()
    
    
    return test

    

