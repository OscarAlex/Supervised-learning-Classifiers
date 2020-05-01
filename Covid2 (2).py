# IMPORTAR LIBRERÍAS
import pandas as pd
import glob
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, ZeroPadding2D
from livelossplot import PlotLossesKeras
import matplotlib.pyplot as plt
import seaborn as sns

# Declaración de variables globales
PATH_COVID  = 'C:/Users/Oscar/Documents/Python Scripts/Redes neu/3er ex/covid-chestxray-dataset-master/images'
PATH_PNEUMONIA  = 'C:/Users/Oscar/Documents/Python Scripts/Redes neu/3er ex/chest_xray/test/PNEUMONIA'
PATH_NORMAL  = 'C:/Users/Oscar/Documents/Python Scripts/Redes neu/3er ex/chest_xray/test/NORMAL'
IMAGE_SIZE    = (128, 128)
NUM_CLASSES   = 3
INPUT_SHAPE = (128,128,1)
BATCH_SIZE    = 32  # try reducing batch size or freeze more layers if your GPU runs out of memory
NUM_EPOCHS    = 10


# 1. CARGAR DATOS
#Cargar metadatos del dataset
df = pd.read_csv('C:/Users/Oscar/Documents/Python Scripts/Redes neu/3er ex/covid-chestxray-dataset-master/metadata.csv')
#print(df['finding'].value_counts())
#De la columna 'finding' obtener todas las instancias con Covid
df_cov = df[df['finding'] =='COVID-19']
#df_cov['filename']

#Covid
#Cargar imágenes de Covid del directorio, especificadas por el nombre que está en columna 'filename' de df_cov, en escala de grises
img_list = [cv2.imread(item, cv2.IMREAD_GRAYSCALE) for i in [glob.glob(PATH_COVID+'/%s' % img) for img in df_cov['filename']] for item in i]
#Ajustar todas las imágenes a 800x800
resCov = [cv2.resize(i, IMAGE_SIZE) for i in img_list]
#Convertir lista a array
resCov = np.asarray(resCov)
#Crear lista con label 0 = Covid
Covlab = [0] * len(resCov)
#Imprimir cantidad de imágenes
print("Covid =", len(Covlab))

#Pneumonía
#Cargar imágenes de Neumonía del directorio, en escala de grises
img_list1 = [cv2.imread(item, cv2.IMREAD_GRAYSCALE) for i in [glob.glob(PATH_PNEUMONIA+'/*')] for item in i]
#Borrar elementos para que sean la misma cantidad que Covid
del img_list1[len(resCov):]
#Ajustar las imágenes
resNeu = [cv2.resize(i, IMAGE_SIZE) for i in img_list1]
#Convertir lista a array
resNeu = np.asarray(resNeu)
#Crear lista con label 1 = Pneumonía
Neulab = [1] * len(resNeu)
#Imprimir cantidad de imágenes
print("Pneumonía =", len(Neulab))

#Normal
#Cargar imágenes de personas sanas del directorio, en escala de grises
img_list2 = [cv2.imread(item, cv2.IMREAD_GRAYSCALE) for i in [glob.glob(PATH_NORMAL+'/*')] for item in i]
#Borrar elementos para que sean la misma cantidad que Covid
del img_list2[len(resCov):]
#Ajustar las imágenes
resNorm = [cv2.resize(i, IMAGE_SIZE) for i in img_list2]
#Convertir lista a array
resNorm = np.asarray(resNorm)
#Crear lista con label 2 = Normal
Normlab = [2] * len(resNorm)
#Imprimir cantidad de imágenes
print("Normal =", len(Normlab))


# 2. PREPROCESAR DATOS
print(resCov.shape, resNeu.shape, resNorm.shape)
#Concatenar los arrays de imágenes, orden= Cov, Pneu, Norm
fullArr = np.concatenate((resCov, resNeu, resNorm), axis=0)
#Añade una dimensión a las imágenes
fullArr = fullArr.reshape((fullArr.shape[0], fullArr.shape[1], fullArr.shape[2], 1))
#Obtener el valor máximo del array
maxm = np.amax(fullArr)
#Convertir a float
fullArr= fullArr.astype('float32')/maxm
#Concatenar las listas con labels, orden= Cov, Pneu, Norm
fullLab= Covlab+Neulab+Normlab
#Imprimir dimensiones
#print("Features =", fullArr.shape)
#print("Labels =", len(fullLab))

# Particionar (20%) y mezclar las instancias
x_tr, x_tst, y_tr, y_tst= train_test_split(fullArr, fullLab, test_size=.20)
# Imprimir las dimensiones
#print("X_train =", x_tr.shape)
#print("X_test =", x_tst.shape)
#print("Y_train =", len(y_tr))
#print("Y_test =", len(y_tst))
# Obtener las dimensiones de las imagenes
INPUT_SHAPE = x_tr.shape[1:] 
# Número de clases
NUM_CLASSES = len(np.unique(y_tr))
# Imprimir las dimensiones de las imagenes
print("Imgs =", INPUT_SHAPE)
print("Classes =", NUM_CLASSES)

#%%
# 3. CREAR EL MODELO
# Crear modelo secuencial
model = Sequential()

# Añadir layers al modelo
model.add(ZeroPadding2D(padding=(2,2), input_shape=INPUT_SHAPE))
model.add(Conv2D(8, (3,3), activation='tanh', kernel_initializer='he_uniform'))
model.add(MaxPool2D((2, 2)))

model.add(ZeroPadding2D((2,2)))
model.add(Conv2D(8, (3,3), activation='tanh', kernel_initializer='he_uniform'))
model.add(MaxPool2D((2,2)))
#model.add(Dropout(0.2))
model.add(ZeroPadding2D((2,2)))
model.add(Conv2D(16, (3,3), activation='tanh', kernel_initializer='he_uniform'))
model.add(MaxPool2D((2,2)))

model.add(ZeroPadding2D((2,2)))
model.add(Conv2D(16, (3,3), activation='tanh', kernel_initializer='he_uniform'))
model.add(MaxPool2D((2,2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(100, activation='tanh', kernel_initializer='he_uniform'))

model.add(Dense(NUM_CLASSES, activation='softmax'))

#from keras.optimizers import Adam
#adam= Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

# Compilar el modelo usando las métricas de accuracy para medir el rendimiento
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# 4. ENTRENAR EL MODELO
model.fit(x_tr, y_tr, epochs=12, batch_size=60, validation_data=(x_tst, y_tst), callbacks=[PlotLossesKeras()], verbose=0)
#model.fit(x_tr, y_tr, epochs=10, batch_size=BATCH_SIZE, validation_split=.20, callbacks=[PlotLossesKeras()], verbose=0)


# 5. EVALUAR EL MODELO
loss, acc = model.evaluate(x_tst, y_tst, verbose=0)
print('Accuracy: %.3f' % acc)
print('Loss: %.3f' % loss)
predictions = model.predict_classes(x_tst)
cm = confusion_matrix(y_tst,predictions)
print(cm)
print(classification_report(y_tst,predictions))
#sns.heatmap(cm, center=True)
#plt.show()

# Overfitting = better performance on training set than the test set
# Underfitting = poor performance on both training and test set