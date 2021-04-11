import numpy as np
import random as ra
A = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,1,2,3,4,5])
#A=list(map(int,input().split()))
B= np.array([1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25])
#B=list(map(int,input().split()))
A=A.reshape(5,5)
B=B.reshape(5,5)
At=A.transpose()
#print (At)
#print(B)
#print (A*B)
def grad(x):
    return 2*(A*x+B)*At
print(grad(5))