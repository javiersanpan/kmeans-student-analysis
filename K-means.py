import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from random import randrange as rr
import random

def genData(n,seed=1,m=False): #Genera data linearmente separable por una pendiente (m)
    np.random.seed(seed)
    if(m==False):
    	m= (rr(20)+1)/10
    X = np.random.random((n, 2))
    y = ((X[:,1]/X[:,0])>m)*1
    X = (X-0.5)*2
    return [X,y]

def graphPoints(X,y): #Grafica los puntos y los puntos proyectados
    for i in range(0,X.shape[0]):
        p = X[i,0:2]
        #p2 = Xp[i,0:2]            
        if(y[i]):
            plt.scatter(p[0],p[1],color="red", s=4)
        else:
            plt.scatter(p[0],p[1],color="black", s=4)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.show()

def createXClases(n,seed=1,m=False): #Modificar para crear dos tipos de puntos
    if(m==False):
    	m= ((rr(20)+1)/10)**0.5
    X = np.random.random((n, 2))
    y = ((X[:,1]/X[:,0])>m)*1
    X = (X-0.5)*2
    return [X,y]

def distanceX1AndX2(X2,X):
    listDistance = np.zeros((len(X),len(X2)))
    for i in range(len(X2)):
        for j in range(len(X)):
            distance = ( (X2[i][0]-X[j][0])**2 + (X2[i][1]-X[j][1])**2 )**0.5
            listDistance[j][i] = distance
    return listDistance

def distanceX1OrX2(yy,X,listdistanceX1AndX2):
    i = 0
    listClasification = np.zeros(len(X))
    for j in range(0,len(X)):
        if(listdistanceX1AndX2[j][i]>listdistanceX1AndX2[j][i+1]):
            #listdistanceX1AndX2[j][i+1] = 0
            listClasification[j] = 1  #Modificar de acuerdo al tipo de punto
        else:
            #listdistanceX1AndX2[j][i+1] = 1
            listClasification[j] = 0  #Modificar de acuerdo al tipo de punto
    return listClasification

[X,y] = genData(20)
graphPoints(X,y)

[X2,yy] = createXClases(2)
graphPoints(X2,yy)

listdistanceX1AndX2 = distanceX1AndX2(X2,X)

distanceX1OrX2 = distanceX1OrX2(X2,X,listdistanceX1AndX2)

graphPoints(X,distanceX1OrX2)

