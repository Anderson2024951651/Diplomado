import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from yellowbrick.cluster import SilhouetteVisualizer
from yellowbrick.datasets import load_nfl

#DEFINICIÓN DE FUNCIONES:

#FUNCIÓN "1"
#FUNCIÓN QUE IMPRIME LA INERCIA Y LOS INDICADORES DAVIES BOULDIN Y SILHOUETTE PARA UN RANGO DE CLUSTERS:
#SIRVE PARA DETERMINAR QUE NÚMERO DE CLUSTER ES EL ÓPTIMO
def evalua_clusters(df, rango_clusters):
    resultados={'clusters': [],'davies_bouldin': [],'Silhouette': [],'inercia': []}
    for k in range(rango_clusters[0], rango_clusters[1] + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, max_iter=10)
        kmeans.fit(df)
        labels = kmeans.labels_
        inercia = kmeans.inertia_
        dav_score = davies_bouldin_score(df, labels)
        sil_score = silhouette_score(df, labels)
        resultados['clusters'].append(k)
        resultados['davies_bouldin'].append(dav_score)
        resultados['Silhouette'].append(sil_score)
        resultados['inercia'].append(inercia)
    resultados_df=pd.DataFrame(resultados)
    print(resultados_df)
    plt.figure()
    plt.plot(resultados_df['clusters'], resultados_df['inercia'], marker='o', color='g', label='Inercia')
    plt.title('Evaluación de Clusters: Inercia', fontsize=14)
    plt.xlabel('Número de Clusters', fontsize=12)
    plt.ylabel('Valor', fontsize=12)
    plt.legend()
    plt.show()
    plt.figure()
    plt.plot(resultados_df['clusters'], resultados_df['davies_bouldin'], marker='o', color='b', label='Davies-Bouldin')
    plt.title('Evaluación de Clusters: Davies-Bouldin', fontsize=14)
    plt.xlabel('Número de Clusters', fontsize=12)
    plt.ylabel('Valor', fontsize=12)
    plt.legend()
    plt.show()
    plt.figure()
    plt.plot(resultados_df['clusters'], resultados_df['Silhouette'], marker='o', color='r', label='Silhouette')
    plt.title('Evaluación de Clusters: Silhouette', fontsize=14)
    plt.xlabel('Número de Clusters', fontsize=12)
    plt.ylabel('Valor', fontsize=12)
    plt.legend()
    plt.show()
    return resultados_df

#FUNCIÓN "2"
#FUNCIÓN PARA GRAFICAR CENTROIDE E IMPRIMIR METRICAS DE INERCIA, SILHOUETTE Y DAVIES BOULDIN:
#SIRVE PARA EL REPORTE DE RESULTADOS UNA VEZ ELEGIDO EL NÚMERO DE CLUSTER ÓPTIMO
def grafica_centroides_y_métricas(df, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(df)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    inertia = kmeans.inertia_
    sil_score = silhouette_score(df, labels)
    dav_score = davies_bouldin_score(df, labels)
    plt.figure(figsize=(8, 6))
    plt.scatter(df['A'], df['B'], c=labels, cmap='viridis', marker='o', s=50, alpha=0.7, label='Datos')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='X', label='Centroides')
    plt.title(f'Centroides y Clustering con KMeans ({n_clusters} Clusters)', fontsize=14)
    plt.xlabel('A', fontsize=12)
    plt.ylabel('B', fontsize=12)
    plt.legend()
    plt.show()
    print(f'Inercia para {n_clusters} clusters: {inertia}')
    print(f'Silhouette Score para {n_clusters} clusters: {sil_score}')
    print(f'Davies-Bouldin Score para {n_clusters} clusters: {dav_score}')

#FUNCIÓN "3"
#FUNCIÓN PARA GRAFICAR CENTROIDE E IMPRIMIR METRICAS DE INERCIA, SILHOUETTE Y DAVIES BOULDIN:
#SIRVE PARA EL REPORTE DE RESULTADOS UNA VEZ ELEGIDO EL NÚMERO DE CLUSTER ÓPTIMO
def imprimir_centroides_y_metricas(df,n_clusters):
    kmeans = KMeans(n_clusters=n_clusters,random_state=42)
    kmeans.fit(df)
    print(kmeans.cluster_centers_)
    print(kmeans.labels_)
    print(kmeans.inertia_)
