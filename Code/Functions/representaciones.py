import numpy as np
import pandas as pd
import re
import os
import math
import numpy as np
import pandas as pd
import joblib
import warnings
import funciones
from funciones import *
import time
import matplotlib.pyplot as plt

#### 9 MUESTRAS:
# avg_spectrum : Todos los espectros en una misma gráfica
# avg_spectrum_all : Todos los espectros cada uno en una gráfica

# plot_clusters : Clústers (3x3)
# plot_clusters_class : Clúster (3x3) con clasificacion
# plot_clusters_fila : Fila con un solo clúster (eliges muestra)
# plot_clusters_fila_comparativa : Fila con varios Clusters (eliges muestras)

#### 8 MUESTRAS
# avg_spectrum_all_8muestras : Todos los espectros cada uno en una gráfica
# plot_clusters_8muestras_2col : Clusters (2colX4row)
# plot_clusters_8muestras_4col :  Clusters (4colX2row)

muestras_raman = ["Moderna 1", "Moderna 2", "Moderna 3", "Moderna 4", "Moderna 5", "Moderna 6", "Moderna 7", "Arq 235", "Arq 885"]

muestras_raman_8 = ["Moderna 1", "Moderna 2", "Moderna 3", "Moderna 4", "Moderna 5", "Moderna 6", "Arq 235", "Arq 885"]

dic_col = {
    'azul': '#2D9DEA',
    'amarillo': '#FFD700',
    'rojo': '#d62728',
    'verde': '#1EB440',
    'morado': '#9467bd',
    'naranja' : '#FFCC79',
    'azul claro': 'skyblue',
    'gris': 'gray',
    'rosa': 'pink'
    # Agrega más colores según sea necesario
}

dic_cluster = {
   'Cluster_KMeans' : 'K-Means',
   'Cluster_KMeans_RF' : 'K-Means y Random Forest',
   'Cluster_Agg_RF' : 'AGNES y Random Forest',
   'Cluster_Birch_RF' : 'Birch y Random Forest',
   "Cluster_KMeans_kNN" : 'K-Means y k-NN',
   'Cluster_Agg_kNN': 'AGNES y k-NN',
   'Cluster_Birch_kNN': 'Birch y k-NN'
}

colores = list(dic_col.values())

#################### Pintar espectros medios #############################
def avg_spectrum (data,
                  muestras = muestras_raman, 
                  path="Datos", titulo = "Espectro medio de las muestras", nombre_archivo = "No"):
  #data = pd.concat(dataframes, axis=0)
  #data = create_df(csv_files, path)

  plt.figure(figsize=(12, 4))

  for i, muestra in enumerate(muestras):
    data_muestra = data[data.index.get_level_values('Muestra') == f'{muestra}']
    wavenumber = data_muestra.columns.get_level_values(0).astype(float)
    # Calculamos las intensidades medias para cada numero de onda
    mean_intensity = data_muestra.mean(axis=0)
    color = colores[i % len(colores)]

    # Representa el espectro medio con el color correspondiente
    plt.plot(wavenumber, mean_intensity, label=f'{muestra}', color=color)

  # Etiquetas y título para el gráfico
  plt.xlabel("Número de Onda / cm$^{-1}$ ")
  plt.ylabel('Intensidad (u.a.)')
  plt.title(f'{titulo}')

  plt.legend()
  plt.grid(True)

  if nombre_archivo != "No":
    plt.savefig(f'Graficas/Espectros/{nombre_archivo}_media.png')

  plt.show()

###################### Espectros medios TODOS #####################

def avg_spectrum_all(data, colores = colores, nombre_archivo = "No", muestras = muestras_raman):
    #data = pd.concat(dataframes, axis=0)
    fig, axes = plt.subplots(3, 3, figsize=(14, 8))
    # Itera sobre las muestras y representa cada una en su subplot
    for i, muestra in enumerate(muestras):
        data_muestra = data[data.index.get_level_values('Muestra') == f'{muestra}']
        wavenumber = data_muestra.columns.get_level_values(0).astype(float)
        # Calcula las intensidades medias para cada número de onda
        mean_intensity = data_muestra.mean(axis=0)

        # Usa el color correspondiente de la lista de colores
        color = colores[i % len(colores)]

        # Calcula la posición del subplot en la matriz de subplots
        fila = i // 3
        columna = i % 3

        # Representa el espectro medio en el subplot correspondiente
        axes[fila, columna].plot(wavenumber, mean_intensity, label=f'Muestra {muestra}', color=color)
        axes[fila, columna].set_xlabel("Número de Onda / cm$^{-1}$")
        axes[fila, columna].set_ylabel('Intensidad (u.a.)')
        axes[fila, columna].set_title(f'Muestra {muestra}')
        axes[fila, columna].set_ylim(0, 6500)
        axes[fila, columna].grid(True)

    # Ajusta el espaciado entre subplots
    plt.tight_layout()

    #plt.savefig('Graficas/medias_todos.png')
    if nombre_archivo != "No":
      plt.savefig(f'Graficas/Espectros/{nombre_archivo}_medias.png')

    # Muestra los subplots
    plt.show()


#################### Plot numeros de clusters #############################
# Plotea los diagramas que muestran el Número de Espectros de cada Cluster para
# cada muestra.
def plot_clusters (data, muestras = muestras_raman, tipo_cluster = "Cluster_KMeans", color=dic_col['azul'], nombre_archivo = "No"):

  # Crear subplots con el diseño deseado
  fig, axes = plt.subplots(3, 3, figsize=(10, 9))

  todos_clusters = sorted(data[tipo_cluster].unique())
  indices_cluster = np.arange(len(todos_clusters))

  # Realizamos para cada muestra
  for i, muestra in enumerate(muestras):
      # Separamos los espectros de la muestra
      data_muestra = data[data.index.get_level_values("Muestra") == muestra]

      # Contamos el núvero de veces de espectros correspondientes al clúster
      cluster_counts = data_muestra[tipo_cluster].value_counts().reindex(todos_clusters, fill_value=0)

      # Creamos un barplot
      row = i // 3
      col = i % 3
      bar_cluster = axes[row, col].bar(indices_cluster, cluster_counts, width=0.9, label=tipo_cluster, color=color)

      # Editamos el título para cada muestra:
      axes[row, col].set_title(f"Muestra {muestra}")
      axes[row, col].set_xticks(indices_cluster)
      axes[row, col].set_xticklabels(todos_clusters)
      axes[row, col].set_xlabel("Clúster")
      axes[row, col].set_ylabel("Número de Espectros")

  # Añado un título
  fig.suptitle(dic_cluster[tipo_cluster])

  # Ajustar espaciado entre subplots
  plt.tight_layout()

  # Mostrar los barplots
  #plt.savefig('Graficas/clustering_KMeans.png')
  if nombre_archivo != "No":
      plt.savefig(f'Graficas/Clusters/{nombre_archivo}_{tipo_cluster}_clust.png')

  plt.show()



#################### Plot numeros de clusters Clasificacion #############################
#Plotea los diagramas que muestran el Número de Espectros de cada Cluster para
# cada muestra.
def plot_clusters_class (data, muestras = muestras_raman, tipo_cluster = "Cluster_KMeans_RF", color=dic_col['azul'], nombre_archivo ="No"):

  # Crear subplots con el diseño deseado
  fig, axes = plt.subplots(3, 3, figsize=(10, 9))

  todos_clusters = sorted(data[tipo_cluster].unique())
  indices_cluster = np.arange(len(todos_clusters))

  # Realizamos para cada muestra
  for i, muestra in enumerate(muestras):
      # Separamos los espectros de la muestra
      data_muestra = data[data.index.get_level_values("Muestra") == muestra]

      # Contamos el núvero de veces de espectros correspondientes al clúster
      cluster_counts = data_muestra[tipo_cluster].value_counts().reindex(todos_clusters, fill_value=0)

      # Creamos un barplot
      row = i // 3
      col = i % 3
      if (muestra == "Arq 235" or muestra =="Arq 885" ):
        bar_cluster = axes[row, col].bar(indices_cluster, cluster_counts, width=0.9, label=tipo_cluster, color=dic_col['rojo'])
      else: 
        bar_cluster = axes[row, col].bar(indices_cluster, cluster_counts, width=0.9, label=tipo_cluster, color=color)
      # Editamos el título para cada muestra:
      axes[row, col].set_title(f"Muestra {muestra}")
      axes[row, col].set_xticks(indices_cluster)
      axes[row, col].set_xticklabels(todos_clusters)
      axes[row, col].set_xlabel("Clúster")
      axes[row, col].set_ylabel("Número de Espectros")

  # Añado un título
  fig.suptitle(dic_cluster[tipo_cluster])

  # Ajustar espaciado entre subplots
  plt.tight_layout()

  if nombre_archivo != "No":
      plt.savefig(f'Graficas/Clusters/{nombre_archivo}_{tipo_cluster}_class.png')

  plt.show()



######################## Plot en una Fila ###################


def plot_clusters_fila (data, muestras = muestras_raman, tipo_cluster = "Cluster_KMeans", color=dic_col['azul'], nombre_archivo = "No"):

  # Crear subplots con el diseño deseado
    fig, axes = plt.subplots(1, len(muestras), figsize=(4*len(muestras),5))

    todos_clusters = sorted(data[tipo_cluster].unique())
    indices_cluster = np.arange(len(todos_clusters))
    

    # Realizamos para cada muestra
    for i, muestra in enumerate(muestras): ## NECESARIO CAMBIAR
        # Separamos los espectros de la muestra
        data_muestra = data[data.index.get_level_values("Muestra") == muestra]

        # Contamos el núvero de veces de espectros correspondientes al clúster
        cluster_counts = data_muestra[tipo_cluster].value_counts().reindex(todos_clusters, fill_value=0)
        todos_clusters = cluster_counts.index
  
        bar_cluster = axes[i].bar(indices_cluster, cluster_counts, width=0.9, label=tipo_cluster, color=color)

        # Editamos el título para cada muestra:
        axes[i].set_title(f"Muestra {muestra}")
        #axes[i].set_xticks(indices_cluster)
        axes[i].set_xticklabels(todos_clusters)
        axes[i].set_xlabel("Clúster")
        axes[i].set_ylabel("Número de Espectros")

    # Añado un título
    fig.suptitle(dic_cluster[tipo_cluster])

    # Ajustar espaciado entre subplots
    plt.tight_layout()

    # Mostrar los barplots
    if nombre_archivo != "No":
      plt.savefig(f'Graficas/Clusters/{nombre_archivo}_{tipo_cluster}_fila.png')

    plt.show()

######################## Plot en una Fila Comparativa###################

def plot_clusters_fila_comparativa(data, muestras, tipos_cluster, colores=None, nombre_archivo = "No"):
    # Configurar colores predeterminados si no se proporcionan
 
    # Configurar el diseño de subplots
    fig, axes = plt.subplots(1, len(muestras), figsize=(5*len(muestras),6), sharex=True)
    for ax in axes:
        ax.grid(False)
        
    for i, muestra in enumerate(muestras):
        # Filtrar los datos por muestra
        data_muestra = data[data.index.get_level_values("Muestra") == muestra]

        # Obtener todos los clusters únicos
        todos_clusters = np.unique(data[tipos_cluster].values)

        # Ajustar la posición de las barras para cada tipo de cluster
        bar_width = 0.2
        positions = np.arange(len(todos_clusters))

        for j, tipo_cluster in enumerate(tipos_cluster):
            # Contar el número de veces de espectros correspondientes al tipo de clúster
            cluster_counts = data_muestra[tipo_cluster].value_counts().reindex(todos_clusters, fill_value=0)

            # Ordenar los clusters por sus valores en el eje y (número de espectros)
            sorted_clusters = sorted(todos_clusters, key=lambda x: -cluster_counts[x])

            # Crear el gráfico de barras con los clusters ordenados
            axes[i].bar(positions + j * 0.3, cluster_counts[sorted_clusters], width=0.3, label=f"{tipo_cluster}", color=colores[j])

        # Añadir etiquetas y leyenda al primer subplot
        axes[i].set_title(f"{muestra}")
        axes[i].set_ylabel("Número de Espectros")

    # Añadir etiquetas y leyenda al eje x (común para todos los subplots)
    axes[1].set_xlabel("Clúster")
    axes[0].set_xlabel("Clúster")
    axes[2].set_xlabel("Clúster")
    axes[1].legend(loc='lower center', bbox_to_anchor=(0.5, 1.05),  ncol=1)

    #plt.xticks(positions + 0.3 * (len(tipos_cluster) - 1) / 2, todos_clusters)

    # Ajustar diseño y mostrar el gráfico
    plt.tight_layout()

    if nombre_archivo != "No":
      plt.savefig(f'Graficas/Clusters/{nombre_archivo}_comparativa_fila.png')
    plt.show()


###################### 8 MUESTRAS #####################

def avg_spectrum_all_8muestras(data, colores = colores, nombre_archivo="No", muestras=None):
    if muestras is None:
        muestras = muestras_raman_8

    fig, axes = plt.subplots(4, 2, figsize=(10, 8))

    # Itera sobre las muestras y representa cada una en su subplot
    for i, muestra in enumerate(muestras):
        data_muestra = data[data.index.get_level_values('Muestra') == f'{muestra}']
        wavenumber = data_muestra.columns.get_level_values(0).astype(float)

        # Calcula las intensidades medias para cada número de onda
        mean_intensity = data_muestra.mean(axis=0)

        # Usa el color correspondiente de la lista de colores
        color = colores[i % len(colores)]

        # Calcula la posición del subplot en la matriz de subplots
        fila = i % 4
        columna = i // 4

        # Representa el espectro medio en el subplot correspondiente
        axes[fila, columna].plot(wavenumber, mean_intensity, label=f'Muestra {muestra}', color=color)
        axes[fila, columna].set_xlabel("Número de Onda / cm$^{-1}$")
        axes[fila, columna].set_ylabel('Intensidad (u.a.)')
        axes[fila, columna].set_title(f'Muestra {muestra}')
        axes[fila, columna].set_ylim(0, 6500)
        axes[fila, columna].grid(True)

    # Ajusta el espaciado entre subplots
    plt.tight_layout()

    if nombre_archivo != "No":
        plt.savefig(f'Graficas/Espectros/{nombre_archivo}_medias_8.png')

    # Muestra los subplots
    plt.show()
    plt.show()


def plot_clusters_class_8muestras_2col (data, muestras = muestras_raman_8, tipo_cluster = "Cluster_KMeans_RF", color=dic_col['azul'], nombre_archivo ="No"):

  # Crear subplots con el diseño deseado
  fig, axes = plt.subplots(4, 2, figsize=(8, 12))

  todos_clusters = sorted(data[tipo_cluster].unique())
  indices_cluster = np.arange(len(todos_clusters))

  # Realizamos para cada muestra
  for i, muestra in enumerate(muestras):
      # Separamos los espectros de la muestra
      data_muestra = data[data.index.get_level_values("Muestra") == muestra]

      # Contamos el núvero de veces de espectros correspondientes al clúster
      cluster_counts = data_muestra[tipo_cluster].value_counts().reindex(todos_clusters, fill_value=0)

      # Creamos un barplot
      row = i // 2
      col = i % 2
      if (muestra == "Arq 235" or muestra =="Arq 885" ):
        bar_cluster = axes[row, col].bar(indices_cluster, cluster_counts, width=0.9, label=tipo_cluster, color=dic_col['rojo'])
      else: 
        bar_cluster = axes[row, col].bar(indices_cluster, cluster_counts, width=0.9, label=tipo_cluster, color=color)
      # Editamos el título para cada muestra:
      axes[row, col].set_title(f"Muestra {muestra}")
      axes[row, col].set_xticks(indices_cluster)
      axes[row, col].set_xticklabels(todos_clusters)
      axes[row, col].set_xlabel("Clúster")
      axes[row, col].set_ylabel("Número de Espectros")

  # Añado un título
  fig.suptitle(dic_cluster[tipo_cluster])

  # Ajustar espaciado entre subplots
  plt.tight_layout()

  if nombre_archivo != "No":
      plt.savefig(f'Graficas/Clusters/{nombre_archivo}_{tipo_cluster}_class_8muestras_2col.png')

  plt.show()

def plot_clusters_class_8muestras_4col(data, muestras=muestras_raman_8, tipo_cluster="Cluster_KMeans_RF", color=dic_col['azul'], nombre_archivo="No"):

    # Crear subplots con el diseño deseado
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    todos_clusters = sorted(data[tipo_cluster].unique())
    indices_cluster = np.arange(len(todos_clusters))

    # Realizamos para cada muestra
    for i, muestra in enumerate(muestras):
        # Separamos los espectros de la muestra
        data_muestra = data[data.index.get_level_values("Muestra") == muestra]

        # Contamos el número de veces de espectros correspondientes al clúster
        cluster_counts = data_muestra[tipo_cluster].value_counts().reindex(todos_clusters, fill_value=0)

        # Creamos un barplot
        fila = i % 2
        columna = i // 2

        # Verificamos si la muestra es "Arq 235" o "Arq 885" para cambiar el color a rojo
        color_actual = dic_col['rojo'] if muestra in ["Arq 235", "Arq 885"] else color

        bar_cluster = axes[fila, columna].bar(indices_cluster, cluster_counts, width=0.9, label=tipo_cluster, color=color_actual)

        # Editamos el título para cada muestra:
        axes[fila, columna].set_title(f"Muestra {muestra}")
        axes[fila, columna].set_xticks(indices_cluster)
        axes[fila, columna].set_xticklabels(todos_clusters)
        axes[fila, columna].set_xlabel("Clúster")
        axes[fila, columna].set_ylabel("Número de Espectros")

    # Añado un título
    fig.suptitle(dic_cluster[tipo_cluster])

    # Ajustar espaciado entre subplots
    plt.tight_layout()

    if nombre_archivo != "No":
        plt.savefig(f'Graficas/Clusters/{nombre_archivo}_{tipo_cluster}_class_8muestras_4col.png')

    plt.show()


    #################### Plot Clusters Muestra 7 #############################
def plot_clusters_SP(data, titulo = "Distribución de Clústers para diferentes Métodos",
        nombre_archivo = "Clusters_SP"):
    
  # Lista de tipos de clustering 
  clusters_names = ["Cluster_KMeans", "Cluster_Agglomerative","Cluster_Spectral", "Cluster_Birch" ] 
  cluster_title =["K-Means", "AHC","Spectral", "Birch" ]

  # Defino los colores para cada tipo de clustering

  cluster_colors_kmeans = {0: "lemonchiffon", 1: "firebrick", 2: "white"} 
  cluster_colors_agglomerative = {0: "lemonchiffon", 1: "firebrick", 2: "white"} 
  cluster_colors_spectral = {0: "lemonchiffon", 1: "white", 2: "firebrick"}  
  cluster_colors_birch = {0: "lemonchiffon", 1: "firebrick", 2: "white"}  


  # Crea una figura con subplots en una fila y tres columnas
  fig, axs = plt.subplots(2, 2, figsize=(10, 8))

  # Obtengo los datos X e Y para representar los espectros de la muestra
  data["X"] = data.index.get_level_values(0)
  data["Y"] = data.index.get_level_values(1)

  # Para cada tipo de clustering, crea un subplot
  for i, cluster_column in enumerate(clusters_names):
      axs[i // 2, i % 2].set_facecolor("lightgray")
      for cluster in data[cluster_column].unique():
          subset = data[data[cluster_column] == cluster]
          if cluster_column == "Cluster_KMeans":
              axs[i // 2, i % 2].scatter(subset["X"], subset["Y"], marker=",", color=cluster_colors_kmeans[cluster], s=13)
          elif cluster_column == "Cluster_Agglomerative":
              axs[i // 2, i % 2].scatter(subset["X"], subset["Y"], marker=",", color=cluster_colors_agglomerative[cluster], s=13)
          elif cluster_column == "Cluster_Birch":
              axs[i // 2, i % 2].scatter(subset["X"], subset["Y"], marker=",", color=cluster_colors_birch[cluster], s=13)
          elif cluster_column == "Cluster_Spectral":
              axs[i // 2, i % 2].scatter(subset["X"], subset["Y"], marker=",", color=cluster_colors_spectral[cluster], s=12)
          

      # Añado los ejes, el título y la leyenda
      axs[i // 2, i % 2].set_title(cluster_title[i])
      axs[i // 2, i % 2].set_xlabel("X")
      axs[i // 2, i % 2].set_ylabel("Y")

      # Ajusto los límites de las imágenes
      axs[i // 2, i % 2].set_xlim(data["X"].min(), data["X"].max())
      axs[i // 2, i % 2].set_ylim(data["Y"].min(), data["Y"].max())

  # Ajusta el espaciado entre subplots para que no se superpongan
  plt.tight_layout()
  fig.suptitle(titulo)

  #plt.savefig('Graficas/comparación_clustering7.png')

  # Muestra la figura
  plt.show()