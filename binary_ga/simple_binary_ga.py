"""
A Binary Genetic Algorithm 
Finds the maximum value of the function f(x) = x^2
"""
import numpy as np

def start_population(lim_min, lim_max, population_size):
    """
    Choose a random population from the search space

    Params:

        - lim_min: min possible value for x
        - lim_max: max possible value for x
        - population_size: number of individuals for the population

    Returns:
        - population
    """

    # choose individuals from search space
    random_individuals = np.random.randint(lim_min, lim_max, size=population_size)
    # transform each integer into a binary number with 5 positions
    population = [list("{0:05b}".format(x)) for x in random_individuals]
    return population

def select_parents(p, p_size):
    """

    Params:

        -p: population
        -p_size: population size

    """

    selected = list()
    for i in range(p_size):

        ixs = np.random.choice(np.arange(p_size), size=2, replace=True)
        x1, fit_s1 = fit(p[ixs[0]])
        x2, fit_s2 = fit(p[ixs[1]])
        if (fit_s1 >= fit_s2):
            selected.append(p[ixs[0]])
        else:
            selected.append(p[ixs[1]])
  
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

def fit(i):
    x = int(''.join(i), 2)
    fx = function(x)
    return x, fx

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

def function(x):
    """
    A simple quadratic function
    """
    return x**2

def GA(generations, population, prct_cx, prct_mt, pop_size):
    """
    Run the Binary Genetic Algorithm to find the value of x that has the maximum value on the 
    function f(x) = x², with 0<= x <=31 

    Params:
        - generations: number of times the algorithm will run
        - population: initial population
        - prct_cx: percentage used in the crossover (60% to 90%)
        - prct_mt: percentege used in mutation (~1%)
        - pop_size: population size

    Returns:
        - best
    """

    history = list()
    best_i = -np.inf
    best_x = None
    for g in range(generations):

        selected_parents = select_parents(population, pop_size)
        ixs, childs = crossover(selected_parents, prct_cx, p_size=pop_size)
        childs = mutation(childs, prct_mt)
        population = update_population(population, ixs, childs)
        for p in population:
            x, best_local = fit(p)
            if (best_local >= best_i):
                best_i = best_local
                best_x = x
        
        history.append(best_x)
    return history


pop_size = 4
population = start_population(0, 32, pop_size)
hist = GA(4, population, 60, 1, pop_size)

print (hist)