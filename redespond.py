"""
Created on Sun Mar  8 15:09:17 2020
RED NEURONAL :O
@author: Oscar
"""

import numpy as np
#Leer archivo y meterlo en una matriz
with open('dataset.csv') as archivo:
    df= [l.strip().split(",") for l in archivo]
#Borrar primera fila
df= np.delete(df, (0), axis=0)

#Hacer lista con las clases y matriz con las features
labels= []
features= np.zeros((190,6))
index= 0
for i in df:
    fila2 = df[index]
    features[index,:] = fila2[0:6]
    labels.append(fila2[-1])
    index+= 1
#print(features)
#print(labels)

#Normalizar features
from sklearn.preprocessing import MinMaxScaler
scaler= MinMaxScaler()
Normfeat= scaler.fit_transform(features)

#Particionar dataset
from sklearn.model_selection import train_test_split
F_train, F_test, L_train, L_test= train_test_split(Normfeat, labels, test_size=.20)

#Clasificador
from sklearn.neural_network import MLPClassifier
mlp= MLPClassifier(hidden_layer_sizes= (150,150), activation= 'tanh', 
                   max_iter= 1500, solver= 'sgd', learning_rate_init= .009, 
                   tol= .000001, alpha= .0008, verbose= 1)
mlp.fit(F_train, L_train)

#Accuracy
scores= mlp.score(F_test, L_test)
print("Accuracy")
print(scores)

#import matplotlib.pyplot as plt
#plt.ylabel('cost')
#plt.xlabel('iterations')
#plt.title("Loss curve")
#plt.plot(mlp.loss_curve_)
#plt.show()

from livelossplot import PlotLossesKeras
mlp.fit(F_train, L_train,
          epochs=10,
          validation_data=(F_test, L_test),
          callbacks=[PlotLossesKeras()],
          verbose=0)

