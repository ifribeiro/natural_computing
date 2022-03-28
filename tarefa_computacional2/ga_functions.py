import numpy as np
import functions

def selection(pop, scores, k=3):
    """
    Perform the selection by tournament
    """
    # first random selection
    selection_ix = np.random.randint(len(pop))
    for ix in np.random.randint(0, len(pop), k-1):
        # perform a tournament
        if scores[ix][0] < scores[selection_ix][0]:
            selection_ix = ix
    return pop[selection_ix]

def crossover(p1, p2):
    """
    Perform the aritimethic crossover
    """
    beta = np.random.uniform()
    c1_p1 = [a*beta for a in p1]
    c1_p2 = [a*(1-beta) for a in p2] 
    c1 = [a1+a2 for a1, a2 in zip(c1_p1, c1_p2)]

    c2_p1 = [(1-beta)*a for a in p1]
    c2_p2 = [beta*a for a in p2]
    c2 = [a1+a2 for a1, a2 in zip(c2_p1, c2_p2)]

    return [c1,c2]

def mutation(p, r_mut, alpha):
    """
    Perform the gaussian mutation
    """
    new_p = []
    for p_i in p:
        pmutate = p_i.flatten()
        for i in range(len(pmutate)):
            if np.random.rand() < r_mut:
                pmutate[i] = np.random.normal(loc=pmutate[i], scale=alpha, size=None)
        pmutate = pmutate.reshape(p_i.shape)
        new_p.append(pmutate)
    return new_p

def GA_NN(layers=None, X_train=None, X_test=None, y_train=None, y_test=None, n_iter=100, n_pop=50, r_mut=0.05, alpha=0.01, r_decr=0.001, encoder=None, loss=None, lograte=-1):
    """
    Genetic Algorithm for training a NN

    Params:
        - layers: NN architeture
        - X_train: training samples
        - X_test: test samples
        - y_train: labels for training samples
        - y_test: labels for test samples
        - n_iter: epochs of training
        - n_pop: population size
        - r_mut: crossover rate
        - alpha: mutation step
        - r_decr: decaying of alpha
        - encoder: one hot encoder
        - loss: loss function name (sigmoid, relu or softmax)
        - lograte: rate to print logs
    """
    
    # initialize the population    
    pop = functions.initialize_population(layers, n_pop)
    # training and validation scores
    best_eval = np.inf
    best_eval_v = np.inf
    # loss and accuracy history
    hist_loss_train = []
    hist_loss_vali = []
    hist_acc_train = []
    hist_acc_vali = []
    # encodes labels
    y_true = encoder.inverse_transform(y_train).flatten()
    y_true_vali = encoder.inverse_transform(y_test).flatten()
    n_flag = int(n_iter/2)
    cnt_loss = 0

    for gen in range(n_iter):
        # evaluate all candidates in the population on training and validation
        scores = [functions.eval_individual(p, layers, X_train, y_train, y_true, loss=loss, encoder=encoder) for p in pop]
        scores_vali = [functions.eval_individual(p, layers, X_test, y_test, y_true_vali, loss=loss, encoder=encoder) for p in pop]
        
        # checks best candidate on training and validation
        for i in range(n_pop):
            if scores[i][0] < best_eval:
                best, best_eval = pop[i], scores[i][0]
                acc_training = scores[i][1]
            if (scores_vali[i][0] < best_eval_v):
                best_eval_v = scores_vali[i][0]
                acc_valid = scores_vali[i][1]
        if (gen%lograte== 0) and (lograte>0):
            print ("#{} | loss_train:{:.2f} | loss_vali:{:.2f} | acc_train:{:.2f} | acc_vali:{:.2f}".format(gen, best_eval, best_eval_v, acc_training, acc_valid))
        
        if (len(hist_loss_train)>=1 and (best_eval == hist_loss_train[-1])):
            cnt_loss +=1
        else:
            cnt_loss=0        
        if (cnt_loss==n_flag): 
            print ("Early stopping (network stop improving)")
            break

        # saves scores and accuracy      
        hist_loss_train.append(best_eval)
        hist_loss_vali.append(best_eval_v)
        hist_acc_train.append(acc_training)
        hist_acc_vali.append(acc_valid)  
              
        # select parantes
        selected = [selection(pop, scores) for _ in range(n_pop)]
        # create the next generation    
        children = list()
        for i in range(0, n_pop, 2):
            # get selected parents in pairs
            p1, p2 = selected[i], selected[i+1]
            # crossover and mutation
            for c in crossover(p1, p2):
                # mutation
                c = mutation(c, r_mut, alpha)
                # store for next generation
                children.append(c)
        alpha -= r_decr
        # replace population
        pop = children

    metrics = [hist_loss_train, hist_loss_vali, hist_acc_train, hist_acc_vali]
    return best, metrics