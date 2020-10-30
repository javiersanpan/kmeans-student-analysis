# TC-1002-5

# Proyecto final de Herramientas computacionales: el arte de la anlítica

## Integrantes del grupo 5
*En orden alfabético*

Gutierrez Aldrete José de Jesús (ShoyChoy)

López Cruz Diana Sareli (dianasareli)

Ojeda Angulo Carlos Noel (Carlos-Ojeda)

Sánchez Panduro Francisco Javier (javiersanpan)

Saucedo Cruz José Rodrigo (SauceRSC)

## Objetivo de la clase

Esta semana Tec está diesñada para desarrollar el análisis de datos en ingeniería y ciencias, así como la identificación de variables. 
Asimismo, se desarrolla el emprendimiento innovador, y la inovación. 

## Objetivo del proyecto

Se usó un dataset, ****insertar***

## Proyecto

Se hizo una encuesta con 7 tipos de personas diferentes, en la que las personas que participaban decían que tanto pertenecían a cada clase en un ranking de 1 a 5. 

Este dataset se lee con la librería de Pandas. 

Se divide en matriz de caractéristicas y vector de clases. Con esto, se probaron varios métodos de reducción de dimensiones, y se generan graficas con cada uno de ellos. 

Las graficas generadas son: 


![Isomap](https://github.com/javiersanpan/TC-1002-5/blob/master/Imagenes/Isomap.jpeg)

![PCA](https://github.com/javiersanpan/TC-1002-5/blob/master/Imagenes/PCA.jpeg)

![MDS](https://github.com/javiersanpan/TC-1002-5/blob/master/Imagenes/MDS.jpeg)

Y se seleccionó el ***insertar seleccionada**, esto fue porque al intentar hacer la prueba con los tres, este es el que mostraba las clases de manera más clara, esto se puede observar cuando se separan por colores:

![ISOMAP](https://github.com/javiersanpan/TC-1002-5/blob/readme/Imagenes/ISOMAP-C.png)

![PCA](https://github.com/javiersanpan/TC-1002-5/blob/master/Imagenes/PCA-C.jpeg)

*Se omite MDS porque existe un error al dividir entre 0.*

Adicionalmente, se programó el kmeans, y se grafican los puntos y los centroides en cada interacción, y de esta manera se ve como evoluciona la posición de estos centroides. 

