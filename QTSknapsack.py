import time
import numpy as np
from itertools import compress
from functools import reduce
from math import pi,sqrt,cos,sin

N = 10  # Neighbourhood size
n_items = 250
MaxWeight = 10.0
MinWeight = 1.0

# Generate array of possible items
# Real valued weights
items = np.random.uniform(low=MinWeight, high=MaxWeight, size=(n_items,))
#print(items)

# Integer valued weights
# items = np.random.randint(low=MinWeight,high=MaxWeight, size=(n_items,))
# print(items)

# Bag maximum capacity
C = reduce(lambda x,y : x+y, items) / 2
# Profits array for each item. The item index corresponds to its associated profit value
profits = np.vectorize(lambda x: x + 5)(items)

print("Item search space: %s" % items)
print("Item profits %s" % profits)
print("Bag capacity: %i" % C)


def calculate_weights(items, solution):
    """Calculate the weight of a solution"""
    return reduce(lambda x,y: x+y, compress(items, solution), 0)



def measure(qindividuals):
    """Consecutive measures on the qbits in order to generate a classical solution"""
    return np.vectorize(lambda x,y : 1 if (x > np.power(y,2)) else 0)\
                        (np.random.rand(n_items), qindividuals[:, 1])


def gen_nbrs(qindividuals, N):
    """Apply n measures on the qbits to generate classical solutions"""
    neighbours = [np.array(measure(qindividuals)) for i in range(N)]
    return neighbours


def adjust_solution(solution, C):
    """Implements the repair method in order to respect the problem constraints.
       Lamarckian greedy repair, i.e. consecutive deletion of selected items until
       the constraints are  satisfied
    """
    itemsSelected = solution.nonzero()[0]
    weight = calculate_weights(items, solution)
    while (weight > C):
        r = np.random.randint(0,itemsSelected.shape[0]-1)
        j = itemsSelected[r]
        solution[j] = 0
        weight = weight - items[j]
        itemsSelected = np.delete(itemsSelected, r)
    return solution


def adjust_neighbours(vizinhos, C):
    """Make the necessary adjustments to keep the generated solutions valid"""
    new_neighbours = [np.array(adjust_solution(vizinho, C)) for vizinho in vizinhos]
    return new_neighbours


def new_best_fit(new_solution, best_fit):
    """Compare the new solution with the current best"""
    if (calculate_weights(profits, new_solution) > calculate_weights(profits, best_fit)):
        return new_solution
    return best_fit


def find_best_worst(neighbours):
    """Find the best and worst solution within a neighbourhood"""
    tmp = [np.array(calculate_weights(profits, vizinho)) for vizinho in neighbours]
    return (neighbours[np.argmax(tmp)], neighbours[np.argmin(tmp)])

#Atualiza a população de q-bits. A lista tabu é considerada dentro da função, durante o loop de aplicação das rotações.
#A aplicação da porta quânica em um q-bit k qualquer é proibido de ser aplicado(tabu) caso ambos os bit k da melhor e pior
#solução da iteração sejam, concomitantemente, 0 ou 1(Função xnor = 1 então é tabu).
def updateQ(worst_sol, best_sol, qindividuals):
    """Update the qbits population applying the quantum gate on each qbit.
       The movement is not made for those qbits on the tabu list"""
    theta = 0.01*pi
    
    for i in range(n_items):
        mod_sinal = best_sol[i] - worst_sol[i]
        # Check on which quadrant kth qbit is located and modify theta accordingly
        if (qindividuals[i, 0]*qindividuals[i, 1] < 0) : mod_sinal *= -1
                
        Ugate = np.array([[cos(mod_sinal*theta), -sin(mod_sinal*theta)],
                          [sin(mod_sinal*theta),  cos(mod_sinal*theta)]])  # Rotation matrix
        qindividuals[i, :] = np.dot(Ugate, qindividuals[i, :])
    return qindividuals


qindividuals = np.zeros((n_items, 2))
qindividuals.fill(1 / sqrt(2))
solution = measure(qindividuals)

best_fit = solution

i = 0
NumIter = 1000

print("Bag weight limit: %f" % C)
print("Number of iterations: %i" % NumIter)
print("Number of items: %i" % n_items)
print("Initial weight (without repair): %f" % calculate_weights(items, best_fit))

best_fit = adjust_solution(best_fit, C)

print("Initial weight (with repair): %f" % calculate_weights(items, best_fit))
print("Initial profit (with repair): %f" % calculate_weights(profits, best_fit))

start_time = time.time()

while i < NumIter:
    i = i + 1
    neighbours = gen_nbrs(qindividuals, N)
    neighbours = adjust_neighbours(neighbours, C)
    (best_solution, worst_solution) = find_best_worst(neighbours)
    best_fit = new_best_fit(best_solution, best_fit)
    qindividuals = updateQ(best_solution, worst_solution, qindividuals)

print("Running time : %.2f seconds" % (time.time() - start_time))
print("Best solution profit %f" % calculate_weights(profits, best_fit))
print("Best solution weight: %f" % calculate_weights(items, best_fit))