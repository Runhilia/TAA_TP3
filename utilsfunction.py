import random
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def etiquetage(y_train_, pourcentage):
    list_contain = []
    for i in range(len(y_train_)):
        if y_train_[i] not in list_contain:
            list_contain.append(y_train_[i])
            pass
        if random.random() > pourcentage:
            y_train_[i] = np.nan
    return y_train_

def pertinence(X_train, y_train):
    # Récupérer la partie labélisé
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    index_lab = np.where(~np.isnan(y_train))
    X_train_lab = X_train[index_lab]
    y_train_lab = y_train[index_lab]
    
    # Calcule du score de fisher
    score_fisher = fisher(X_train_lab, y_train_lab)
    
    index_non_lab = np.where(np.isnan(y_train))
    X_train_non_lab = X_train[index_non_lab]
    y_train_non_lab = y_train[index_non_lab]
    
    score_laplacien = laplacien(X_train_non_lab, y_train_non_lab)
    
    return score_fisher / score_laplacien
    
        
def fisher(X_train_lab, y_train_lab):    
    # Calcule du score de fisher
    nb_class = len(np.unique(y_train_lab))
    effectif = np.zeros(nb_class)
    effectif = np.array([np.sum(y_train_lab == i) for i in range(nb_class)])
    
    # Calcul de la moyenne
    moyenne = np.zeros((nb_class, X_train_lab.shape[1]))
    for i in range(nb_class):
        moyenne[i] = np.mean(X_train_lab[y_train_lab == i], axis=0)
        
    moyenne_global = np.mean(X_train_lab, axis=0)
    
    # Calcul de l'écart-type 
    sigma = np.zeros((nb_class, X_train_lab.shape[1]))
    for i in range(nb_class):
        sigma[i] = np.std(X_train_lab[y_train_lab == i], axis=0)
        
    fisher = np.zeros(X_train_lab.shape[1])
    for i in range(X_train_lab.shape[1]):
        fisher[i] = np.sum(effectif * (moyenne[:,i] - moyenne_global[i])**2) / np.sum(effectif * sigma[:,i]**2)
    return fisher
    
def laplacien(X_train_non_lab, y_train_non_lab):
    
    score_lap = np.zeros(X_train_non_lab.shape[1])
    for v in range(X_train_non_lab.shape[1]):
        v_values = X_train_non_lab[:,v]
        variance = np.var(v_values)
        
        S = np.zeros((len(v_values), len(v_values)))
        numerator = np.zeros(len(v_values))
        for i in range(len(v_values)):
            for j in range(len(v_values)):
                diff = v_values[i] - v_values[j]
                S[i,j] = np.exp(-(diff**2) / 10)
                numerator += S[i,j] * (diff**2)
        
        score_lap[v] = np.sum(numerator / variance)
        
    return score_lap
        
def plot_efficacite_courbe(X_train, X_test, y_train, y_test, score, step = 5):
    sorted_index = np.argsort(-score)
    X_train = X_train[:,sorted_index]
    X_test = X_test[:,sorted_index]
    
    n_feature = []
    acc_list = []
    for num_var in range(step, X_train.shape[1]+1, step):
        X_train_sub = X_train[:,:num_var]
        X_test_sub = X_test[:,:num_var]
        
        mlp = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000)
        mlp.fit(X_train_sub, y_train)
        y_pred = mlp.predict(X_test_sub)
        acc_list.append(accuracy_score(y_test, y_pred))
        n_feature.append(num_var)
        
    plt.plot(n_feature, acc_list)
    plt.xlabel('Nombre de variables')
    plt.ylabel('Accuracy')
    plt.title('Efficacité du modèle en fonction du nombre de variables')
    
        
    
    
    
    
    
