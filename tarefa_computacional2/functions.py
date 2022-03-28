from matplotlib import markers
import numpy as np
import pandas as pd
import neural_network as nn
from keras.layers import Dense
from keras.models import Sequential
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import matplotlib.pyplot as plt
import os

def eval_individual(individual, net_layers, x_train, y_train, y_true, loss='bce', encoder=None):
    """
    Evaluates the a individual
    """
    A = x_train
    lim_w = net_layers[0].W.shape[1]
    lim_wo = net_layers[-1].W.shape[1]
    # forward step
    for i in range(len(net_layers)):
        layer = net_layers[i]        
        # output layer
        if i == len(net_layers)-1:
            layer.W = individual[i][:, :lim_wo]            
            layer.b = individual[i][:, lim_wo:]
            A = layer.feedforward(A)
        else:
            layer.W = individual[i][:, :lim_w]
            layer.b = individual[i][:, -1:]
            A = layer.feedforward(A)
    # loss defined
    if (loss=='bce'): # binary crossentropy
        cost = 1/m * np.sum(nn.logloss(y_train, A))
    elif(loss=='mse'): 
        cost = nn.mse(y_train, A)
    elif(loss=='cce'): # categorical cross entropy
        cost = nn.CategoricalCrossentropy(A.T, y_train.T)
    
    # predict 
    y_pred = predict(x_train, net_layers, individual)
    y_pred = encoder.inverse_transform(y_pred).flatten()
    acc = accuracy_score(y_true, y_pred)
    return cost, acc

def set_weights(weights, net_layers):
    """
    Set new weights to the network
    """
    lim_w = net_layers[0].W.shape[1]
    lim_wo = net_layers[-1].W.shape[1]
    for i in range(len(net_layers)):
        layer = net_layers[i]        
        if i == len(net_layers)-1:
            layer.W = weights[i][:, :lim_wo]            
            layer.b = weights[i][:, lim_wo:]
        else:
            layer.W = weights[i][:, :lim_w]
            layer.b = weights[i][:, -1:]

def predict(x_train, layers, weights=None):
    """
    Make predictions
    """
    A = x_train
    if (weights!=None):
        set_weights(weights, layers)
    for layer in layers:
        A = layer.feedforward(A)
    return A

def run_ga_experiments(model, n_tests, nn, exp_name="", args=[], save=False):
    """
    Run GA experiments
    """
    
    loss_training = []
    loss_vali = []
    acc_training = []
    acc_valid = []
    for i in range(n_tests):
        print ("Running exp {}".format(i))
        # hold out
        X_train, y_train, X_test, y_test, encoder = get_samples(args[0], args[1])

        best, metrics = model(layers=nn, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                            n_iter=args[2], n_pop=args[3], r_mut=args[4], alpha=args[5],r_decr=args[6], encoder=encoder,
                            loss=args[7])
        loss_training.append(metrics[0])
        loss_vali.append(metrics[1])

        acc_training.append(metrics[2])
        acc_valid.append(metrics[3])

    if save:
        try:
            os.makedirs("results/{}".format(exp_name))
        except:
            pass
        # save metrics
        np.save("results/{}/losses_training.npy".format(exp_name), np.array(loss_training))
        np.save("results/{}/losses_vali.npy".format(exp_name), np.array(loss_vali))
        np.save("results/{}/accs_training.npy".format(exp_name), np.array(acc_training))
        np.save("results/{}/accs_vali.npy".format(exp_name),np.array(acc_valid))
    
    return loss_training, loss_vali, acc_training, acc_valid



def run_pso_experiments(model, n_tests, nn, exp_name="", args=[], norm=False, save=False):
    """
    Run PSO experiments   
    """    
    
    loss_training = []
    loss_vali = []
    acc_training = []
    acc_valid = []
    for i in range(n_tests):
        print ("Running exp {}".format(i))

        X_train, y_train, X_test, y_test, encoder = get_samples(args[0], args[1], norm=norm)

        best, metrics = model(layers=nn, X_treino=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                            num_iters=args[2], swarm_size=args[3], inertia=args[4], pa=args[5], ga=args[6], encoder=encoder)
        loss_training.append(metrics[0])
        loss_vali.append(metrics[1])

        acc_training.append(metrics[2])
        acc_valid.append(metrics[3])
    if save:
        try:
            os.makedirs("results/{}".format(exp_name))
        except:
            pass

        # save metrics
        np.save("results/{}/losses_training.npy".format(exp_name), np.array(loss_training))
        np.save("results/{}/losses_vali.npy".format(exp_name), np.array(loss_vali))
        np.save("results/{}/accs_training.npy".format(exp_name), np.array(acc_training))
        np.save("results/{}/accs_vali.npy".format(exp_name),np.array(acc_valid))
    return loss_training, loss_vali, acc_training, acc_valid

def initialize_population(layers, n, init='uniform'):
    """
    Initialize population
    """
    population = []
    for i in range(n):
        network = []
        for layer in layers:
            neurons, inputs =  layer.shpW
            if init=='uniform':
                W = np.random.uniform(low=-1.0, high=1.0, size=neurons*inputs).reshape(neurons, inputs)
            else:
                W = np.random.randn(neurons, inputs)
            b = np.zeros((neurons, 1))
            p = np.concatenate([W,b], axis=1)
            network.append(p)
        population.append(network)
    return population

def load_dataset(name=""):
    """
    Loads a dataset
    """
    if name == 'iris':
        iris_data = pd.read_csv("iris.data", header=None)
        X = iris_data.values[:, :-1].astype('float')
        Y = iris_data.values[:, -1]
    elif name == 'wine':
        wine = pd.read_csv("wine.data", header=None)
        features = np.arange(1,14)
        X = wine[features].values
        Y = wine[0].values
    elif name=='breast':
        breast = pd.read_csv("breast_cancer.csv")
        breast_columns = breast.columns
        X = breast[breast_columns[2:-1]].values
        Y = breast['diagnosis'].values
    return X,Y

def get_samples(X, Y, test_size=0.15, norm=True):
    """
    Samples for GA and PSO
    """

    X_train, X_test, y_train, y_teste = train_test_split(X, Y, test_size=test_size)
    if norm:
        X_train = MinMaxScaler().fit_transform(X_train)
        X_test = MinMaxScaler().fit_transform(X_test)
    
    X_train = X_train.T
    X_test = X_test.T

    encoder = OneHotEncoder()
    y_hot_train = encoder.fit_transform(y_train.reshape(-1,1)).toarray()
    y_hot_test = encoder.fit_transform(y_teste.reshape(-1,1)).toarray()

    return X_train, y_hot_train, X_test, y_hot_test, encoder

def run_nn_experiments(n, exp_name="", args=[], save=False):
    """
    Run the baseline experiments
    """
    loss_training = []
    loss_vali = []
    acc_training = []
    acc_valid = []
    
    for i in range(n):
        print ("Running exp ", i)
        X_train, X_test, y_train, y_test = train_test_split(args[0], args[1], test_size=0.01)
        input_shape = X_train.shape[1]
        out_size = len(np.unique(y_train))
        X_train = MinMaxScaler().fit_transform(X_train)
        X_test = MinMaxScaler().fit_transform(X_test)
        enc = OneHotEncoder()
        y_train_hot = enc.fit_transform(y_train.reshape(-1, 1)).toarray()
        model = Sequential()
        model.add(Dense(args[2], input_shape=(input_shape,), activation='relu'))
        model.add(Dense(out_size, activation='softmax'))
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

        h = model.fit(X_train, y_train_hot, batch_size=10, epochs=150, validation_split=0.15, verbose=False)
                
        loss_training.append(h.history['loss'])
        loss_vali.append(h.history['val_loss'])
        acc_training.append(h.history['accuracy'])
        acc_valid.append(h.history['val_accuracy'])
    
    if save:        
        try:
            os.makedirs("results/{}".format(exp_name))
        except:
            pass    

        # save metrics
        np.save("results/{}/losses_training.npy".format(exp_name), np.array(loss_training))
        np.save("results/{}/losses_vali.npy".format(exp_name), np.array(loss_vali))
        np.save("results/{}/accs_training.npy".format(exp_name), np.array(acc_training))
        np.save("results/{}/accs_vali.npy".format(exp_name), np.array(acc_valid))
    return loss_training, loss_vali, acc_training, acc_valid

##### Plotting #####
def plot_metric(vali, train, figname="", ylabel="",labels=['Validação', 'Treinamento'], savefig=False):
    fig, ax = plt.subplots(figsize=(6,4))
    x = np.arange(len(vali))
    ax.plot(x, vali, label=labels[0])
    ax.plot(x, train, label=labels[1])
    ax.set_xlabel("Épocas")
    ax.set_ylabel("{}".format(ylabel))
    plt.legend()
    plt.tight_layout()
    plt.show()
    if savefig:
        plt.savefig("figuras/{}.pdf".format(figname))
    plt.clf()
    plt.close()

def plot_comparacao(dict_valores, size_valores=50, ylabel="", figname="", savefig=False):
    fig, ax = plt.subplots(figsize=(5,3.5))
    x = np.arange(size_valores)
    ls = ['-', '-.', ':']
    for i, k in enumerate(dict_valores):
        ax.plot(x, dict_valores[k], label=k, ls=ls[i])
    ax.set_xlabel("Épocas", fontdict={"fontsize":13})
    ax.set_ylabel("{}".format(ylabel),fontdict={"fontsize":13})
    plt.legend()
    plt.tight_layout()
    if savefig:
        plt.savefig("figuras/{}.pdf".format(figname))
    plt.show()
    plt.clf()
    plt.close()  