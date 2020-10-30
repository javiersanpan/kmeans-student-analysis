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
    y = np.zeros((n))+50
    X = (X-0.5)*2
    return [X,y]

def graphPoints(X,y,centroide=True): #Grafica los puntos y los puntos proyectados
    if(centroide):
        plt.scatter(X[:,0],X[:,1],c=y)
    else:
        plt.scatter(X[:,0],X[:,1],c=y,marker='x', s=200)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)

def crearCentroides(n,seed=2,m=False): # Crear tipos de puntos para clasificar
    np.random.seed(seed)
    if(m==False):
    	m= ((rr(20)+1)/10)**0.5
    X = np.random.random((n, 2))
    X = (X-0.5)*2
    y = np.arange(n)
    return [X,y]

def distancias(X2,X): # Crear matriz con la distancia de cada punto al punto clasificador
    listDistance = np.zeros((len(X),len(X2)))
    for i in range(len(X2)):
        for j in range(len(X)):
            distance = ( (X2[i][0]-X[j][0])**2 + (X2[i][1]-X[j][1])**2 )**0.5
            listDistance[j][i] = distance
    return listDistance

def clasificador(yy,X,listDistanceX1AndX2): # Crear matriz con el tipo de punto clasificador mas cercano
    listClasification = np.zeros(len(X))
    for j in range(0,len(X)):
        listClasification[j] = posMenor(listDistanceX1AndX2[j],0,0,0)
    return listClasification

def posMenor(listDistanceX1AndX2,j,i,posi): # Regresa el tipo de punto al que pertenece dicho punto
    if (j == 0):
        return posMenor(listDistanceX1AndX2,j+1,listDistanceX1AndX2[0],0)
    elif ((j < len(listDistanceX1AndX2)) and (i > listDistanceX1AndX2[j])):
        return posMenor(listDistanceX1AndX2,j+1,listDistanceX1AndX2[j],j)
    elif (j < len(listDistanceX1AndX2)):
         return posMenor(listDistanceX1AndX2,j+1,i,posi)
    else:
        return posi
    
def centroides(X,listClasificador,yy): # Regresa los centroides con sus respectivas coordenadas
    XX = np.zeros((len(yy),2))
    for i in range(len(yy)):
        XX[i] = (sum(X[np.where(listClasificador==i)])/len(np.where(listClasificador==i)[0]))
    return XX

def iniciar():
    [X2,yy] = crearCentroides(5)
    graphPoints(X2,yy,False)
    
    listDistance = distancias(X2,X)
    listClasificador = clasificador(X2,X,listDistance)
    graphPoints(X,listClasificador)
    plt.show()
    
    run = True
    while (run):
        X2Viejo = X2
        X2 = centroides(X,listClasificador,yy)
        graphPoints(X2,yy,False)
        listDistance = distancias(X2,X)
        listClasificador = clasificador(X2,X,listDistance)
        graphPoints(X,listClasificador)
        plt.show()
        
        if (np.linalg.norm(X2Viejo) == np.linalg.norm(X2)):
            run = False
            
[X,y] = genData(200)
graphPoints(X,y)
plt.show()
iniciar()