import functions
import numpy as np


def initialize_velocities(layers, n):
    """
    Initialize the velocities
    """
    velocities = functions.initialize_population(layers, n)
    return velocities

def update_particles_pos(particles, velocities):
    """
    Update position of particles
    """
    particles_ = list()
    n_layers = len(particles[0])
    for p,v in zip(particles, velocities):
        p_layers = [p[i]+v[i] for i in range(n_layers)]
        particles_.append(p_layers)
    return particles_

def calc_momentum(velocities, inertia):
    """
    Calculate the mommentum of particles
    """
    m = list()
    n_layers = len(velocities[0])
    for v in velocities:
        m_layers = [v[i]*inertia for i in range(n_layers)]
        m.append(m_layers)
    return m

def calc_accelerations(pa, ga, personal_bests, particles, global_best):
    """
    Calculate the accelaration of particles
    """
    acc_local = list()
    acc_global = list()
    for pbest, p in zip(personal_bests, particles):
        l_layer = list()
        g_layer = list()
        # updates local and global velocities
        for i in range(len(particles[0])):
            shp = p[i].shape
            li = (pbest[i]-p[i])*np.random.rand(shp[0], shp[1])*pa
            gi = (global_best[i]-p[i])*np.random.rand(shp[0], shp[1])*ga
            l_layer.append(li)
            g_layer.append(gi)
        
        acc_local.append(l_layer)
        acc_global.append(g_layer)
        
    return acc_local, acc_global

def update_velocities(m, acc_local, acc_global):
    """
    Update velocities of particles
    """
    velocities = list()
    for m_i, al_i, ag_i in zip(m, acc_local, acc_global):
        vlist = list()
        for i in range(len(m[0])):
            v = m_i[i] + al_i[i] + ag_i[i]
            vlist.append(v)
        velocities.append(vlist)
    return velocities

def PSO_NN(layers=None, X_treino=None, X_test=None, y_train=None, y_test=None, encoder=None, swarm_size=20, inertia=0.5, pa=0.8, ga=0.9, num_iters=100, lograte=-1):
    """
    Particle Swarm Optimization algorithm for training a Neural Network

    Params:
        - layers: NN architeture
        - X_treino: train samples
        - X_test: test samples
        - y_train: labels for training samples
        - y_test: labels for test samples
        - encoder: encoder for labels
        - swarm_size: size of the swarm
        - inertia: inertia value for momentum
        - pa: particle acceleration rate
        - ga: global acclerationg rate
        - num_iters: number of epochs
        - lograte: rate to print logs

    """
    # encoded labels for training and validation samples
    y_true = encoder.inverse_transform(y_train).flatten()
    y_true_v = encoder.inverse_transform(y_test).flatten()

    # initialize particles and velocities
    particles = functions.initialize_population(layers, swarm_size)
    velocities = initialize_velocities(layers, swarm_size)
    
    # initialize personal and global bests
    pbest = [p for p in particles]
    pbest_loss = [np.inf for _ in particles]
    pbest_loss_v = [np.inf for _ in particles]

    gbest_idx = np.argmin(pbest_loss)
    gbest = pbest[gbest_idx]
    
    # initialize fitness 
    gbest_loss, global_acc = functions.eval_individual(gbest, layers, X_treino, y_train,
                                    y_true, loss='cce', encoder=encoder)
    gbest_loss_v, global_acc_v = functions.eval_individual(gbest, layers, X_test, y_test,
                                    y_true_v, loss='cce', encoder=encoder)
    
    loss_train = []
    loss_vali = []
    acc_train = []
    acc_vali  = []

    for i in range(num_iters):
        # evaluate current swarm
        for p_i in range(swarm_size):
            fitness, acc = functions.eval_individual(particles[p_i], layers, X_treino, y_train, 
                                                    y_true, loss='cce', encoder=encoder)
            fitness_v, acc_v = functions.eval_individual(particles[p_i], layers, X_test, y_test,
                                                    y_true_v, loss='cce', encoder=encoder)
            if fitness < pbest_loss[p_i]:
                pbest[p_i] = particles[p_i]
                pbest_loss[p_i] = fitness
                pbest_loss_v[p_i] = fitness_v
        
        # global best
        if np.min(pbest_loss) < gbest_loss:
            gbest_idx = np.argmin(pbest_loss)
            gbest = pbest[gbest_idx]
            gbest_loss, global_acc = functions.eval_individual(gbest, layers, X_treino, y_train, y_true,
                                        loss='cce', encoder=encoder)
            
            gbest_loss_v, global_acc_v = functions.eval_individual(gbest, layers, X_test, y_test, y_true_v,
                                        loss='cce', encoder=encoder)

        
        # Calculate the momentum
        m = calc_momentum(velocities, inertia)
        # Calculate local and global accelerations
        acc_local, acc_global = calc_accelerations(pa, ga, pbest, particles, gbest)
        # Update the velocities
        velocities = update_velocities(m, acc_local, acc_global)
        # Update the position of particles
        particles = update_particles_pos(particles, velocities)

        loss_train.append(gbest_loss)
        loss_vali.append(gbest_loss_v)
        acc_train.append(global_acc)
        acc_vali.append(global_acc_v)

        if (i%lograte==0 and lograte>0):
            print ("#{} | loss_train:{:.2f}, loss_vali:{:.2f} | acc_train:{:.2f}, acc_vali:{:.2f}".format(i, gbest_loss, gbest_loss_v ,global_acc, global_acc_v))
    metrics = [loss_train, loss_vali, acc_train, acc_vali]
    return gbest, metrics