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

#HOLAHOLAHOLAMUNDOMUNDO

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

def drawVector(lv): #list of vectors
    maxV = 0
    colors = ['orange', 'darkorange', 'lime', 'darkcyan', 'indigo', 'purple', 'mediumspringgreen', 'deeppink']
    for i in range(len(lv)):
        v = lv[i]  
        plt.quiver(0, 0, v[0], v[1], color=colors[i%len(colors)], angles='xy', scale_units='xy', scale=1)
        if(max(v) > maxV):
            maxV = max(v)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.show()

def puntosProyectados(X,v):
    listaPuntos = []
    for i in X:
        listaPuntos.append(projuv(i,v))
    listaPuntos = np.array(listaPuntos)
    return listaPuntos

def crearVector():
    v = np.array([random.random()-0.5, random.random()-0.5])
    return v/la.norm(v)

def projuv(u,v):
    num = np.dot(u,v)
    denom = la.norm(v)**2
    return (num/denom)*v

[X,y] = genData(200)

graphPoints(X,y)

for i in range(20):
    v = crearVector()
    drawVector([v])
    Xp = puntosProyectados(X,v)
    graphPoints(Xp,y)
