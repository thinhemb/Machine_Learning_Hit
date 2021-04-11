import numpy as np
import random as ra
import matplotlib.pyplot as plt
yt=[]
for i in range(100):
    y=2*i+3-ra.random()
    yt.append([i,y])
# a=1
# b=2
a=ra.random()
b=ra.random()
def los(yt,a,b):
    loss=0
    for i in range(100):
        loss=loss + (yt[i][1]-a*yt[i][0]-b)**2
    loss=loss/100
    return loss

def lossab(yt,a,b):
    los1=0
    los2=0
    for i in range(100) :
        los1=los1 +2*(yt[i][1]-a*yt[i][0]-b)*(-yt[i][0])/100
        los2=los2+2*(yt[i][1]-a*yt[i][0]-b)*(-1)/100
    return los1,los2
epoch=100
for i in range(epoch):
    lo=los(yt,a,b)
    losa,losb=lossab(yt,a,b)
    a=a-0.0001*losa
    b=b-0.0001*losb
    print (a,b,lo)
