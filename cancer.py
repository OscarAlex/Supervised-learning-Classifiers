# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 10:37:06 2020

@author: Oscar
"""
#Importar librería numpy
import numpy as np
#Importar librería pandas
import pandas as pd

#Leer archivo csv
df = pd.read_csv('data.csv')

#Obtener el nombre de las columnas
names= df.columns

#Asignar el nombre de la columna de las clases = 'diagnosis'
cl= names[-1]

#Separar las instancias con B de la columna ('diagnosis')
df_max = df[df[cl] =='B']
#Separar las instancias con M de la columna ('diagnosis')
df_min = df[df[cl] =='M']

#Contar los valores de M y B
#M= 212
#df_min[cl].value_counts() 
#B= 357
#df_max[cl].value_counts() 

#Importar clase resample
from sklearn.utils import resample
#Aumentar el número de muestras del menor (M) a 350
df_min_aux= resample(df_min, n_samples= 350)

#Combinar B con las muestras aumentadas de M
df= pd.concat([df_max, df_min_aux])
#Obtener número de filas y columnas
rows, cols= df.shape

#Crear lista para llenar con las clases
labels= []
#Crear matriz para llenar con las features
features= np.zeros((rows,cols-1))
#Indice para el ciclo
index= 0

#Llenar la matriz y lista con los valores del dataframe
for i in range(rows):
    #Obtener la fila
    fila2= df.iloc[index]
    #Llenar la matriz con la instancia del df menos la clase
    features[index,:] = fila2[0:cols-1]
    #Agregar la clase de la fila a la lista
    labels.append(fila2[-1])
    #Actualizar indice
    index+= 1

#Normalizar features
#Importar clase MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
#Crear el modelo de MMS
scaler= MinMaxScaler()
#Normalizar la matriz de features
Normfeat= scaler.fit_transform(features)

#Importancia de las features
#Importar clase ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesClassifier
#Crear el modelo de ETC
etc = ExtraTreesClassifier(n_estimators= 8)
#Ajustar el modelo con las features y labels
etc.fit(features, labels)

#Asignar la importancia de las features
importances = etc.feature_importances_
#print(importances)

#Obtener los índices de menor a mayor de las importancias
indices = np.argsort(importances)[::-1]

#Print the feature ranking
#print("Importances")
#for f in range(features.shape[1]):
#   print("Feature %d (%f)" % (indices[f], importances[indices[f]]))

#Crear matriz para llenar con las features ordenadas por importancia
idx = np.zeros(Normfeat.shape)
#Normfeat[indices[0]]

#Indice para el ciclo
index= 0
#Acomodar por orden de importancia las features
for i in range(len(indices)):
    #Obtiene por columnas y así las mete a la matriz
    idx[:,index]= Normfeat[:,indices[index]]
    #Actualizar indice
    index+=1

#Eliminar las últimas columnas no tan importantes
idx= np.delete(idx, range(8,index), axis= 1)

#Crear lista para meter los nombres de las features más importantes
featI= []
#Obtener los nombres de las features importantes
for i in range(8):
    #Agregar el nombre de la feature de acuerdo al indice
    featI.append(names[indices[i]])

#idx -> 8 features más importantes
#labels -> labels del df
from sklearn.preprocessing import LabelEncoder
idx= idx.astype('float32')
labels = LabelEncoder().fit_transform(labels)

"""
#Calcula la desviación estandar por feature
estdev = np.std([l.feature_importances_ for l in etc.estimators_],
             #axis=0)
import matplotlib.pyplot as plt
#plt.figure()
plt.title("Feature importances")
plt.bar(range(features.shape[1]), importances[indices],
       color="r", yerr=estdev[indices], align="center" )
plt.xticks(range(features.shape[1]), indices)
plt.xlim([-1, features.shape[1]])
plt.show()
"""

#Particionar dataset
#Importar clase train_test_split
from sklearn.model_selection import train_test_split
#Particionar dataset, 80% train y 20% test
F_tr, F_tst, L_tr, L_tst= train_test_split(idx, labels, test_size=.20)


#from tensorflow.keras import Sequential
#from tensorflow.keras.layers import Dense



