"""
Código com funções compartilhadas entre os notebooks
"""

import numpy as np
import pandas as pd

### Funções para os algoritmos ES e DE
def calc_distance(data:np.ndarray, centroids:np.ndarray)->list():
    """
    Compute distancies among data and centroids
    """
    distances = []
    for c in centroids:
        distance = np.sum((data - c) * (data - c), axis=1)
        distances.append(distance)

    distances = np.array(distances)
    distances = np.transpose(distances)
    return distances

def predict(data:np.ndarray, centroids:np.ndarray)-> list():
    """
    Predict clusters based on centroids

    Params:
        -data: data samples
        -centroids: computed centroids

    return:
        - clusters: cluster for each data point
    
    """
    distances = calc_distance(data, centroids)
    clusters = assign_cluster(distances)
    return clusters

def assign_cluster(distances):
    """
    Assign a cluster based on distance
    """
    cluster = np.argmin(distances, axis=1)
    return cluster

def quantization_error(centroids: np.ndarray, labels: np.ndarray, data: np.ndarray) -> float:
    error = 0.0
    for i, c in enumerate(centroids):
        idx = np.where(labels == i)[0]
        dist = np.linalg.norm(data[idx] - c, axis=1).sum()
        error += dist
    return error

def eval_individual(centroids, predict_data, data):
    """
    Avalia um indivíduo da população
    """
    score = quantization_error(centroids, predict_data, data)
    return score

def inicia_populacao(N, bounds, n_clusters, data):
    """
    Params:
        -N: tamanho da população
        -bounds: limites da população criada
        -n_clusters: numero de clusters
        -data: base de dados real
    
    return:
        - u: população de individuos
    """
    
    u = [np.random.uniform(bounds[0], bounds[1], size=n_clusters*data.shape[1]) for _ in range(N)]
    # transforma cada individuo no formato esperado para o cluster (matriz com 3*4 dimensões)
    u = [p.reshape(n_clusters,data.shape[1]) for p in u]
    
    return u


### Funções DE

def reproducao_crossover(U, Fu, CR):
    """
    Cria a população temporaria U_t

    Params:
        - U: população real
        - Fu: parâmetros da mutação da população real
        - CR: taxa de mutação
    return:
        - U_t: população temporária
        - Fy: parâmetros da população temporária
    """
    # inicializaçao da população temporária
    U_t = np.zeros_like(U)
    # indices da população original
    indices = np.arange(0,len(U))
    # vetor os parâmetros da população temporaria
    Fy = []
    for i in range(len(U)):
        # escolha dos parâmetros a, b, c
        abc_ = np.random.choice(indices,replace=False, size=4)
        # seleciona parâmetros diferentes de i
        a, b, c = list(set(abc_).difference([i]))[:3]
        ua = U[a]
        ub = U[b]
        uc = U[c]

        ### calculo do individuo temporario correspondente a U[i]
        rnd1 = np.random.uniform()
        Fy.append(Fu[i])
        # verifica se os parâmetros da população temporária serão atualizados
        if rnd1 < 0.1:
            # calculo do novo parâmetro
            Fy_ = 0.05 + np.random.uniform()*0.35
            # atualiza Fy
            Fy[i] = Fy_
        # calculo do individuo
        y_g = ua + Fy[i]*(ub - uc)
        # realiza crossover entre dois individuos
        y_ = crossover(U[i],y_g,CR)
        U_t[i] = y_        
    return U_t, Fy

def crossover(u, u_t, CR):
    """
    Realiza o crossover entre dois individuos
    
    Params:
        - u: individuo da populacao original
        - u_t: individuo da populacao temporaria
        - CR: taxa de crossover
    """
    # copia o individuo original
    ut = u.copy()
    # indice que garante que ao menos um elemento do vetor
    # veio do vetor temporario
    d = np.random.randint(len(u))
    for j in range(len(u)):
        r = np.random.uniform()     
        # verifica se o crossover deve ser realizado   
        if ((r<CR) or (j==d)):
            ut[j] = u_t[j]
    return ut



### Funções para carregamento das bases de dados

def load_iris():
    data = pd.read_csv('iris.data', header=None)
    targets = np.unique(data[[4]].values)
    return data, targets

def load_wine():
    data = pd.read_csv('wine.data', header=None)
    targets = np.unique(data[[0]].values)
    return data, targets

def load_breast_cancer():
    breast_cancer = pd.read_csv("breast_cancer.csv")
    breast_cancer.drop(['id','Unnamed: 32'], inplace=True, axis=1)
    # valores únicos das labels
    targets_breast = breast_cancer['diagnosis'].unique()

    return breast_cancer, targets_breast