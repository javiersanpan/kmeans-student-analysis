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
    y = ((X[:,1]/X[:,0])>m)*1+20
    X = (X-0.5)*2
    return [X,y]

def graphPoints(X,y,centroide=True): #Grafica los puntos y los puntos proyectados
    if(centroide):
        for i in range(0,X.shape[0]):
            p = X[i,0:2]
            #p2 = Xp[i,0:2]            
            if(y[i]==0):
                plt.scatter(p[0],p[1],color="red", s=4)
            elif(y[i]==1):
                plt.scatter(p[0],p[1],color="blue", s=4)
            elif(y[i]==2):
                plt.scatter(p[0],p[1],color="green", s=4)
            elif(y[i]==3):
                plt.scatter(p[0],p[1],color="brown", s=4)
            elif(y[i]==4):
                plt.scatter(p[0],p[1],color="grey", s=4)
            elif(y[i]==5):
                plt.scatter(p[0],p[1],color="gold", s=4)
            else:
                plt.scatter(p[0],p[1],color="black", s=4)
            
    else:
        for i in range(0,X.shape[0]):
            p = X[i,0:2]
            #p2 = Xp[i,0:2]            
            if(y[i]==0):
                plt.scatter(p[0],p[1],color="red",marker='X', s=4)
            elif(y[i]==1):
                plt.scatter(p[0],p[1],color="blue",marker='X', s=4)
            elif(y[i]==2):
                plt.scatter(p[0],p[1],color="green",marker='X', s=4)
            elif(y[i]==3):
                plt.scatter(p[0],p[1],color="brown",marker='X', s=4)
            elif(y[i]==4):
                plt.scatter(p[0],p[1],color="grey",marker='X', s=4)
            elif(y[i]==5):
                plt.scatter(p[0],p[1],color="gold",marker='X', s=4)
            else:
                plt.scatter(p[0],p[1],color="black",marker='X', s=4)
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        

def crearCentroides(n,seed=2,m=False): # Crear tipos de puntos para clasificar
    np.random.seed(seed)
    if(m==False):
    	m= ((rr(20)+1)/10)**0.5
    X = np.random.random((n, 2))
    y = ((X[:,1]/X[:,0])>m)*1
    X = (X-0.5)*2
    y = [0,1,2,3,4,5]
    return [X,y]

def distancias(X2,X): # Crear matriz con la distancia de cada punto al punto clasificador
    listDistance = np.zeros((len(X),len(X2)))
    for i in range(len(X2)):
        for j in range(len(X)):
            distance = ( (X2[i][0]-X[j][0])**2 + (X2[i][1]-X[j][1])**2 )**0.5
            listDistance[j][i] = distance
    return listDistance

def clasificador(yy,X,listDistanceX1AndX2): # Crear matriz con el punto clasificador mas cercano
    listClasification = np.zeros(len(X))
    for j in range(0,len(X)):
        listClasification[j] = posMayor(listDistanceX1AndX2[j],0,0,0)
    return listClasification

def posMayor(listDistanceX1AndX2,j,i,posi): # Regresa el tipo de punto clasificador al que pertenece dicho punto
    if (j == 0):
        return posMayor(listDistanceX1AndX2,j+1,listDistanceX1AndX2[0],0)
    elif ((j < len(listDistanceX1AndX2)) and (i > listDistanceX1AndX2[j])):
        return posMayor(listDistanceX1AndX2,j+1,listDistanceX1AndX2[j],j)
    elif (j < len(listDistanceX1AndX2)):
         return posMayor(listDistanceX1AndX2,j+1,i,posi)
    else:
        return posi
    

[X,y] = genData(200)
graphPoints(X,y)

[X2,yy] = crearCentroides(3)
graphPoints(X2,yy,False)

listDistance = distancias(X2,X)

listClasificador = clasificador(X2,X,listDistance)

graphPoints(X,listClasificador)
plt.show()
    


