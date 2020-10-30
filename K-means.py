import numpy as np
import matplotlib.pyplot as plt
from random import randrange as rr
import pandas as pd
from sklearn.manifold import Isomap
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import umap

#Se hace la reduccion de 7D a 2D
def reduccion(X):
    X3t = umap.UMAP(X)
    return X3t
'''
    pca = PCA(n_components=2)
    X1t = pca.fit_transform(X)
    
    #Por alguna razon cambia en cada iteracion, por lo que no lo consideramos confiable
    emb = MDS(n_components=2)
    X2t = emb.fit_transform(X)
    
    emb = Isomap(n_components=2)
    X3t = emb.fit_transform(X)
    
    plt.scatter(X1t[:,0],X1t[:,1])
    plt.title('Alumnos dataset, PCA ')
    plt.show()
    plt.scatter(X2t[:,0],X2t[:,1])
    plt.title('Alumnos dataset, MDS')
    plt.show()
    plt.scatter(X3t[:,0],X3t[:,1])
    plt.title('Alumnos dataset, Isomap ')
    plt.show()
    '''

#Se determina K, el numero de clases, mediante Silhouette
def silhouette(X):
    k = 0
    silhouette_avg_past = 0
    range_n_clusters = range(2, 10)

    for n_clusters in range_n_clusters:
        
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        if(silhouette_avg_past < silhouette_avg):
            k=n_clusters
            silhouette_avg_past = silhouette_avg
        
        #Para imprimir el score para cada K y ver cual es el mas alto
        #print("For n_clusters =", n_clusters,"The average silhouette_score is :", silhouette_avg)
        sample_silhouette_values = silhouette_samples(X, cluster_labels)
    
    return k

#Se determina K, el numero de clases,de forma visual
def obtenerK(X):
    Nc = range(2, 10)
    kmeans = [KMeans(n_clusters=i) for i in Nc]
    kmeans
    score = [kmeans[i].fit(X).score(X) for i in range(len(kmeans))]
    score
    plt.plot(Nc,score)
    plt.xlim(0, 10)
    plt.show()

#Grafica los puntos y los puntos proyectados
def graphPoints(X,y,centroide=True,pronostico=True): 
    if(centroide):
        plt.scatter(X[:,0],X[:,1],c=y)
    elif(not(centroide) and not(pronostico)):
        plt.scatter(X[:,0],X[:,1],c=y,marker='2', s=200)
    else:
        plt.scatter(X[:,0],X[:,1],c=y,marker='x', s=200)

# Crear centroides aleatorios para despues reubicarlos
def crearCentroides(n,seed=2,m=False): 
    np.random.seed(seed)
    if(m==False):
    	m= ((rr(20)+1)/10)**0.5
    X = np.random.random((n, 2))
    X = (X-0.5)*2
    y = np.arange(n)
    return [X,y]

# Crear matriz con la distancia de cada punto al punto clasificador
def distancias(X2,X): 
    listDistance = np.zeros((len(X),len(X2)))
    for i in range(len(X2)):
        for j in range(len(X)):
            distance = ( (X2[i][0]-X[j][0])**2 + (X2[i][1]-X[j][1])**2 )**0.5
            listDistance[j][i] = distance
    return listDistance

# Crear matriz con el tipo de punto clasificador mas cercano
def clasificador(yy,X,listDistanceX1AndX2): 
    listClasification = np.zeros(len(X))
    for j in range(0,len(X)):
        listClasification[j] = posMenor(listDistanceX1AndX2[j],0,0,0)
    return listClasification

# Regresa la clase a la que pertenece dicho punto
def posMenor(listDistanceX1AndX2,j,i,posi): 
    if (j == 0):
        return posMenor(listDistanceX1AndX2,j+1,listDistanceX1AndX2[0],0)
    elif ((j < len(listDistanceX1AndX2)) and (i > listDistanceX1AndX2[j])):
        return posMenor(listDistanceX1AndX2,j+1,listDistanceX1AndX2[j],j)
    elif (j < len(listDistanceX1AndX2)):
         return posMenor(listDistanceX1AndX2,j+1,i,posi)
    else:
        return posi

# Regresa los nuevos centroides con sus respectivas coordenadas
def centroides(X,listClasificador,yy): 
    XX = np.zeros((len(yy),2))
    for i in range(len(yy)):
        XX[i] = (sum(X[np.where(listClasificador==i)])/len(np.where(listClasificador==i)[0]))
    return XX

# Hace el Kmeans para 2D
def KMeans2D(k):
    [X2,yy] = crearCentroides(k)
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
    return [X2,yy,X,listClasificador]
def pronostico(X2,yy,X,listClasificador):
    
    X3 =  [[0, 4, 3, 1, 3, 1, 0],
       [1, 2, 3, 4, 1, 2, 3],
       [2, 3, 4, 1, 2, 3, 4],
       [4, 3, 2, 1, 4, 3, 0],
       [0, 3, 5, 4, 3, 5, 4],
       [0, 4, 3, 1, 3, 1, 0]]
    
    #X3t=reduccion(X3)
    emb = Isomap(n_components=2)
    X3t = emb.fit_transform(X3)
    
    listDistance = distancias(X2,X3t)
    listClase = clasificador(yy, X3t,listDistance)
   
    #print(listDistance)
    #print(X3t)
    #print(listClase)
    graphPoints(X3t,listClase,False,False)
    
    #print(X)
    #print(listClasificador)
    graphPoints(X,listClasificador)

    #print(X2)
    #print(yy)
    graphPoints(X2,yy,False)
    
    plt.show()
    

df = pd.read_csv("data.csv")
data = df.values
X = data[:,0:]
y = np.zeros((len(X)))

X=reduccion(X)
obtenerK(X)
K= silhouette(X)
#print("K = ",K) 
[X2,yy,X,listClasificador] = KMeans2D(K)

#pronostico(X2,yy,X,listClasificador)

    