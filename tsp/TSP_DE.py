import math
import numpy as np
import matplotlib.pyplot as plt
import random
import tsplib95

# --- FUNCTIONS ---------------------------------------------------------------+

def funcion(permutacion):
    suma = 0
    for i in range(len(permutacion)-1):
        nodoA = permutacion[i]
        nodoB = permutacion[i+1]
        distancia = tsp.edges[nodoA, nodoB]['weight']
        suma += distancia
    nodoFinal = permutacion[-1]
    nodoInicial = permutacion[0]
    distancia = tsp.edges[nodoFinal, nodoInicial]['weight']
    return suma

def order_crossover(p1, p2):
    start = random.randint(0, len(p1) - 1)
    end = random.randint(start, len(p1))
    sub_cadena = p1[start:end]
    hijo = [0] * len(p1)
    hijo[start:end] = sub_cadena
    p2_sin_subcadena = [valor for valor in p2 if valor not in sub_cadena]
    index = 0
    for i in range(len(hijo)):
        if hijo[i] == 0:
            hijo[i] = p2_sin_subcadena[index]
            index += 1
    return hijo

def mutate_tsp(solution):
    index1, index2 = random.sample(range(len(solution)), 2)
    mutated_solution = solution.copy()
    mutated_solution[index1], mutated_solution[index2] = mutated_solution[index2], mutated_solution[index1]
    return mutated_solution

# --- MAIN -------------------------------------------------------------------+

def main(cost_func, bounds, popsize, mutate, recombination, maxiter, ciudades):
    print(ciudades)
    population = []
    for _ in range(0, popsize):
        ruta_aleatoria = np.random.permutation(ciudades)
        population.append(ruta_aleatoria)
    print(population)

    mejores = []
    peores = []
    promedio = []

    for i in range(1, maxiter+1):
        print('GENERATION:', i)
        gen_scores = []

        for j in range(0, popsize):
            canidates = list(range(0, popsize))
            canidates.remove(j)
            random_index = random.sample(canidates, 1)
            x_1 = population[random_index[0]]
            x_t = population[j]     
            v_donor = mutate_tsp(x_1)
            hijo = order_crossover(v_donor, x_t)
            score_trial  = cost_func(hijo)
            score_target = cost_func(x_t)

            if score_trial < score_target:
                population[j] = hijo
                gen_scores.append(score_trial)
                print('   >', score_trial, hijo)
            else:
                print('   >', score_target, x_t)
                gen_scores.append(score_target)

        gen_avg = sum(gen_scores) / popsize
        gen_best = min(gen_scores)   
        gen_worst = max(gen_scores)                              
        gen_sol = population[gen_scores.index(min(gen_scores))]

        mejores.append(gen_best)
        peores.append(gen_worst)
        promedio.append(gen_avg)

    print('\n\n\n BEST SOLUTION:', gen_sol, '\n')
    return mejores, peores, promedio

def grafica(mejores, peores, promedio, generaciones):
    x = list(range(1, generaciones+1))
    plt.scatter(x, mejores, color='green', label='mejor', s=10)
    plt.plot(x, mejores, color='green')
    plt.scatter(x, peores, color='red', label='peor', s=10)
    plt.plot(x, peores, color='red')
    plt.scatter(x, promedio, color='blue', label='promedio', s=10)
    plt.plot(x, promedio, color='blue')
    plt.legend()
    plt.xlabel('Generaciones')
    plt.ylabel('Aptitud')
    plt.title("Gráfica de convergencia")
    plt.show()

problem = tsplib95.load(r"C:\Users\Brandom\Documents\Bioinspirados\geneticos\tsp\berlin52.tsp")
tsp = problem.get_graph()
print(f"DATOS: {tsp}")

ciudades = tsp.nodes
numeroVariables = len(ciudades)
limiteSuperior = 1
limiteInferior = len(ciudades)
bounds = []

for x in range(numeroVariables):
    bounds.append((limiteInferior, limiteSuperior))

cost_func = funcion                   
popsize = 500                      
mutate = 0.5                        
recombination = 0.7                 
maxiter = 100                       

mejores, peores, promedio = main(cost_func, bounds, popsize, mutate, recombination, maxiter, ciudades)
grafica(mejores, peores, promedio, maxiter)
