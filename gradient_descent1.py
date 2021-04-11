import math
import numpy as np
import matplotlib.pyplot as plt
def f(x):
    return 2*x +5*np.cos(x)
x=5
while abs(f(x))>0.001 :
        x=x-0.011 * f(x)
        print (x)
print(x)
