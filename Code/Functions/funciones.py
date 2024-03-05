
import numpy as np
import pandas as pd
import re
import os
import math
import numpy as np
import pandas as pd
import joblib
import warnings
import time
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, classification_report
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering, Birch
from sklearn.decomposition import PCA


#################### Función para etiquetar las muestras #############################
def nombre_muestra(csv_file):
    file = csv_file.split("_")[0]
    nombre =  re.sub(r'(\D)(\d)', r'\1 \2', file)
    # Pone la primera en mayuscula
    muestra = nombre.capitalize()
    return muestra

#################### Función paracrear los DF #############################

def create_df (csv_files, path = "Datos"):
  import pandas as pd
  dataframes = []

  for csv_file in csv_files:
      df = pd.read_csv(os.path.join(path, csv_file))

      nombre = nombre_muestra(csv_file)
      df.index = pd.MultiIndex.from_tuples([(nombre, X, Y) for X, Y in df.index], names=[ "Muestra", "X", "Y"])

        # Agrega el DataFrame a la lista
      dataframes.append(df)
  data = pd.concat(dataframes)
  return data

#################### Método del codo #############################

from scipy.cluster.hierarchy import linkage
def elbow_method (data, k_values, metodo_clustering, nombre_archivo="No"):
     wcss = [] # Suma de los cuadrados de la distancia
     if metodo_clustering == 'K-Means':
           for k in k_values:
                model = KMeans(n_clusters=k)
                model.fit(data)
                wcss.append(model.inertia_)
                # La WCSS se calcula sumando la distancia al cuadrado de cada punto al centroide 
                # de su cluster asignado y sumando estos 
                # valores para todos los clusters. 
                # Esencialmente, mide la dispersión dentro de cada cluster.
     elif metodo_clustering == 'Agglomerative':
           for k in k_values:
                model = AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='ward')
                labels = model.fit_predict(data)
                linkage_matrix = linkage(data, method='ward')
                wcss.append(sum([d**2 for d in linkage_matrix[:, 2]]))
                #La WCSS se calcula sumando la distancia al cuadrado de cada punto al centroide
                #  del cluster final al que pertenece. La matriz de enlace se utiliza para
                #  determinar a qué cluster final se asigna cada punto.
     else:
        raise ValueError("Método de clustering no válido. Use 'K-Means' o 'Agglomerative'.")
        return
     
     #En resumen, tanto K-Means como Aglomerativo buscan minimizar la dispersión dentro 
     # de los clusters, pero utilizan enfoques diferentes para lograrlo. 
     # K-Means ajusta centroides iterativamente, mientras que Aglomerativo fusiona clusters 
     # jerárquicamente. El cálculo específico de WCSS refleja cómo se mide la dispersión 
     # dentro de los clusters para cada método.

    # Graficar el método del codo
     plt.plot(k_values, wcss, marker='o')
     plt.xlabel('Número de Clusters (K)')
     plt.ylabel('WCSS')
     plt.title(f'Método del Codo utilizando {metodo_clustering}')
     if nombre_archivo != "No":
      plt.savefig(f'Graficas/MetodoCodo/{nombre_archivo}_{metodo_clustering}.png')
     plt.show()


############# Random Forest ############
def class_rf (X, y, test_size=0.2, n_estimators=200):
        # Dividir los datos en conjuntos de train y test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=100)

        # Iniciar y entrenar el clasificador de bosques aleatorios
        rf_classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=100)
        rf_classifier.fit(X_train, y_train)

        # Predecir las etiquetas del conjunto de prueba
        y_pred = rf_classifier.predict(X_test)

        # Evaluar el rendimiento del modelo
        accuracy = accuracy_score(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred)

        return (rf_classifier, accuracy)

############# k-NN ############
def class_knn (X, y, test_size=0.2, n_neighbors=3):
        # Dividir los datos en conjuntos de train y test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=100)

        # Iniciar y entrenar el clasificador de bosques aleatorios
        knn_classifier = KNeighborsClassifier(n_neighbors=3)
        knn_classifier.fit(X_train, y_train)

        # Predecir las etiquetas del conjunto de prueba
        y_pred = knn_classifier.predict(X_test)

        # Evaluar el rendimiento del modelo
        accuracy = accuracy_score(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred)

        return (knn_classifier, accuracy)