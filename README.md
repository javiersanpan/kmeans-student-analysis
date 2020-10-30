# TC-1002-5

# Proyecto final de Herramientas computacionales: el arte de la anlítica

## Integrantes del grupo 5
*En orden alfabético*

Gutierrez Aldrete José de Jesús (ShoyChoy)

López Cruz Diana Sareli (dianasareli)

Ojeda Angulo Carlos Noel (Carlos-Ojeda) *También se le atribuye lo que está a nombre de Carlos sin cuenta de GitHub*

Sánchez Panduro Francisco Javier (javiersanpan)

Saucedo Cruz José Rodrigo (SauceRSC)

## Objetivo de la clase

Esta semana Tec está diseñada para desarrollar el análisis de datos en ingeniería y ciencias, así como la identificación de variables. Asimismo, se desarrolla el emprendimiento innovador, y la inovación. 

## Objetivo del proyecto

Con un dataset brindado, se crea un programa para separar por clases la información, y predecir a cuál pertenecenecerá una persona que responda de nuevo la encuesta. En este proyecto se utiliza un dataset generado por el equipo a través de encuestas, y se pronosticará qué tipo de participante en un equipo es una persona nueva. 

## Proyecto

Se hizo una encuesta con 7 preguntas sobre cargos de equipo diferentes, y hubo 115 participantes. Las personas que participaron contestaron la frecuencia con la que tomaban  cierto cargo en un a escala de 0 a 5. 

Este dataset se lee con la librería de Pandas. 

Se divide en matriz de caractéristicas. Con esto, se probaron varios métodos de reducción de dimensiones, y se generan graficas con cada uno de ellos. 

Las graficas generadas son: 

![Isomap](https://github.com/javiersanpan/TC-1002-5/blob/master/Imagenes/Isomap.jpeg)

![PCA](https://github.com/javiersanpan/TC-1002-5/blob/master/Imagenes/PCA.jpeg)

![MDS](https://github.com/javiersanpan/TC-1002-5/blob/master/Imagenes/MDS.jpeg)

Y se seleccionó ISOMAP, esto fue porque al intentar hacer la prueba con los tres, este es el que mostraba las clases de manera más clara, también se omite MDS, ya que se presenta un error al intentar dividir por 0.

![ISOMAP](https://github.com/javiersanpan/TC-1002-5/blob/master/Imagenes/ISOMAP-C.jpeg)

![PCA](https://github.com/javiersanpan/TC-1002-5/blob/master/Imagenes/PCA-C.jpeg)

Adicionalmente, se programó el kmeans, y se grafican los puntos y los centroides en cada interacción, y de esta manera se ve como evoluciona la posición de estos centroides hasta que llega la medio definitivo. 

K-means es un método de agrupamiento, que hace partición del conjunto de observaciones, en k número de grupos. Cada observación pertenece al grupo cuyo valor medio es más cercano. 

## Cómo usar

1. Corre el script k-means.py
2. Llama a la función `llamarPronostico()`
3. Responde las preguntas que se piden

### Demostración de un pronóstico hecho con el algoritmo

Aquí se muestra como el programa corre al llamarlo. Hace 7 preguntas, y hace un pronostico posicionando a la persona que responde en la gráfica. 

![Prueba](https://github.com/javiersanpan/TC-1002-5/blob/master/Imagenes/test-hecho.png)
