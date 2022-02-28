import numpy as np
import matplotlib.pyplot as plt

def decode(binary, min_x, max_x, l):
    """
    Decode a 22-binary number in a real number between -1.0 and 2.0
    """
    x = min_x + (max_x - min_x)*((binary)/((max_x**l) - 1))
    return x

def encode(x, x_min, x_max, l):
    """
    Encode number between x_min and x_max in a 22-binary number
    """
    int_binary = (x * (x_max**l + x_min) + (x_max**l + x_min))/3
    return int(int_binary)

def start_population(x_min, x_max, l = 22, size=6):
    """
    Start the population
    """
    random_individuals = np.random.uniform(low=x_min, high=x_max, size=size)
    encoded_individuals = [encode(x, x_min, x_max, l) for x in random_individuals]
    # decoded_individuals = [decode(x, x_min, x_max, l) for x in encoded_individuals]
    population = [list("{0:022b}".format(x)) for x in encoded_individuals]    

    return population

def select_parents(population, pop_size):
    selected = list()
    for p in range(pop_size):        
        ixs = np.random.choice(np.arange(pop_size), size=2, replace=True)
        x1, fit_s1 = fx_individual(population[ixs[0]])
        x2, fit_s2 = fx_individual(population[ixs[1]])
        if (fit_s1 >= fit_s2):
            selected.append(population[ixs[0]])
        else:
            selected.append(population[ixs[1]])
    return selected

def choose_parents(prct, population_size):
  """
  Randomly choose parents to be used in the crossover

  Params:
    
    - prct: percentange of population used in the crossover
    - population_size: population size

  Returns:

    - pair_ixs: list of pair of individuals (parents)
  """

  n_individuals = round(prct/100 * population_size)
  pair_ixs = list()
  for i in range(n_individuals//2):
    ixs = np.random.choice(np.arange(population_size), size=2, replace=False)
    pair_ixs.append(ixs)
  
  return pair_ixs

def crossover(p, prct, p_size):
    """
    Perform crossover between parents from population

    Params:

        - p: population
        - prct: percentage of population used in the crossover
        - p_size: population size

    Return:

        - parents_ixs: indices of parents used in the current crossover
        - childs: childs of the crossover
    """

    # choose parents indices
    parents_ixs = choose_parents(prct, p_size)
    childs = list()
    for pi in parents_ixs:
        p1, p2 = pi
        # randomly choose the point of cutoff for the two parents
        cut_ix = np.random.randint(0, high=len(p[p1]))
        
        # 1st child
        c1 = np.concatenate([p[p1][:cut_ix], p[p2][cut_ix:]])
        # 2nd child
        c2 = np.concatenate([p[p2][:cut_ix], p[p1][cut_ix:]])

        childs.append(c1)
        childs.append(c2)

    return parents_ixs, childs


def mutate(v, ix):
    """
    Mutate a individual

    Params:

        -v: vector representatio of the individual
        -ix: indice to be changed

    Return:

        - v_copy: mutated individual
    """
    v_copy = v.copy()
    value = v_copy[ix]
    if value == '1':
        v_copy[ix] = '0'
    else:
        v_copy[ix] = 1
    return v_copy

def mutation(childs, prct):
    """
    Randomly mutates childs from a population

    Params:

        - childs: childs from the population
        - prct: percentage of childs to be mutated

    Returns:

        - childs: 
    """
    # calculates the number of childs to be changed
    mutation_percetage = prct/100
    c_size = len(childs)
    childs_to_change = mutation_percetage * c_size

    # if the number of childs is < 1, no mutation will be performed
    if childs_to_change >= 1:
        # randomily choose childs indices
        childs_ixs = np.random.randint(0,high=c_size,size=int(childs_to_change))
        for cix in childs_ixs:
            ix = np.random.randint(0,high=childs[0].size)
            childs[cix] = mutate(childs[cix],ix)
    return childs

def update_population(population, ixs_remove, childs):
    """
    Update the population after the crossover

    Params:
        
        - population: current individuals in population
        - ixs_remove: individuals to be removed
        - childs: childs to be add in the population
    
    Returns:

        new_population: the new population

    """
    # remove individuals with indices in ixs_remove
    population = np.delete(population, ixs_remove, 0)
    # add childs to the population
    new_population = np.concatenate([population,childs], 0)
    return new_population

def function(x):
    fx = x * np.sin(10*np.pi*x) + 1.0
    return fx

def fx_individual(individual):
    binaryx = int(''.join(individual), 2)
    x_decoded = decode(binaryx,-1, 2, 22)
    fx = function(x_decoded)
    return x_decoded, fx

def GA(generations, population, prct_cx, prct_mt, pop_size):
    """
    Run the Binary Genetic Algorithm to find the value of x that has the maximum value on the 
    function f(x) = x², with 0<= x <=31 

    Params:
        - generations: number of times the algorithm will run
        - population: initial population
        - prct_cx: percentage used in the crossover (60% to 90%)
        - prct_mt: percentage used in the mutation (~1%)
        - pop_size: population size

    Returns:
        - best
    """

    history = list()
    best_i = -np.inf
    best_x = None

    Xf = np.arange(-1,2,0.01)
    Yf = [function(x) for x in Xf]

    for g in range(generations):
        plot_fit_population(population, Xf, Yf, g)
        selected_parents = select_parents(population, pop_size)
        ixs, childs = crossover(selected_parents, prct_cx, p_size=pop_size)
        childs = mutation(childs, prct_mt)
        population = update_population(population, ixs, childs)
        for p in population:
            x, best_local = fx_individual(p)
            if (best_local >= best_i):
                best_i = best_local
                best_x = x
        
        history.append(best_x)
    return history

def plot_fit_population(population, xf, yf, generation):
    X = [fx_individual(i)[0] for i in population]
    fx = [fx_individual(i)[1] for i in population]
    # fx = [function(x) for x in X]
    fig, ax = plt.subplots()
    ax.scatter(X, fx, c='black')
    ax.plot(xf, yf)
    plt.title("Generation {}".format(generation))
    plt.tight_layout()
    plt.savefig("ga_gen_{}.png".format(generation))
    plt.clf()
    plt.close()

    



population = start_population(x_min=-1.0, x_max=2.0, l=22, size=28)
# Xf = np.arange(-1.0,2.0,0.01)
# Yf = [function(x) for x in Xf]

# plot_fit_population(population, Xf, Yf, 0)

hist = GA(25, population, 90, 1, 28)
