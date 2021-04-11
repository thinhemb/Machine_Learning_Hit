import numpy as np
import random as ra
import matplotlib.pyplot as plt
A=np.array([[ 10,11,12,14 ]])
B=np.array([[ 1, 2, 3,4]])
#print(A*B)
plt.plot(A,B,'r*')
plt.axis([1,20,1,20])
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.show()