from numpy import *
def iteration(xx, rr): 
    x, = xx
    r, = rr
    x = r * x * (1 - x)
    return x,