import numpy as np

def ratio_event(y):
    """
    takes an vector of label (0 or 1) and returns the percentage of events
    """
    return 100*sum(y)/y.shape[0]
