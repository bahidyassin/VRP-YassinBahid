import numpy as np
import random
import sys


class Load():
    def __init__(self, load_idx, pickup, dropoff):
        self.idx = load_idx
        self.pickup = pickup
        self.dropoff = dropoff
        
def loadProblemFromFile(path):
    f = open(path, "r")
    loads = []
    for l in f.readlines()[1:]:
        splits = l.replace("\n","").split(" ")
        loads += [Load(int(splits[0]), 
                          tuple(float(s) for s in splits[1].strip("()").split(",")),
                          tuple(float(s) for s in splits[2].strip("()").split(","))
                          )]
    return loads


def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def TotalDistance(Route, dist_matrix, OGToPickup, DropToOG):
    OriginePlusRoute = [0]+Route
    Cost = OGToPickup[Route[0]-1] + dist_matrix[Route[0]-1,Route[0]-1]
    for i in range(1,len(Route)):
        Cost += dist_matrix[Route[i]-1,Route[i-1]-1] + dist_matrix[Route[i]-1,Route[i]-1]
    
    Cost += DropToOG[Route[-1]-1]
    return Cost

def RouteDistance(route, loads):
    total_dist = calculate_distance((0,0), loads[route[0]][0])
    
    for i in range(len(route) - 1):

        total_dist += calculate_distance(loads[route[i]][0], loads[route[i]][1]) 
        total_dist += calculate_distance(loads[route[i]][1], loads[route[i + 1]][0])
    total_dist += calculate_distance(loads[route[-1]][0], loads[route[-1]][1])
    total_dist += calculate_distance((0,0), loads[route[-1]][1])
    return total_dist

def RoutesDistance(Routes, loads):
    RoutesDist = []
    for route in Routes:
        RoutesDist += [RouteDistance(route, loads)]
    return np.sum(RoutesDist)

def total_distance_with_vehicle_penalty(routes, loads, penalty_weight):
    total_dist = 0
    total_unused_capacity_penalty = 0

    for route in routes:

        total_dist += RouteDistance(route, loads)
    num_vehicles_penalty = len(routes) * penalty_weight

    return total_dist + num_vehicles_penalty

def generate_individual(loads, vehicle_capacity):
    individual = []
    load_indices = random.sample(range(len(loads)), len(loads))
    
    while load_indices:
        
        route = []

        while load_indices :
            if len(route) != 0:

                PotentialNewRoute = route.copy() + [load_indices[0]]

                PotentialNewDistance = RouteDistance(PotentialNewRoute,loads)
                

                
                if  PotentialNewDistance > vehicle_capacity:

                    individual.append(route)
                    break
                    
                else:

                    load = load_indices.pop(0)
                    route.append(load)

            else:
                load = load_indices.pop(0)
                route.append(load)
            
    

    if not individual:
        individual.append([])
    individual.append(route)
    return individual

def generate_initial_population(population_size, loads, vehicle_capacity):
    population = []
    
    for _ in range(population_size):
        individual = generate_individual(loads, vehicle_capacity)
        population.append(individual)

    return population

def select_parents(population, tournament_size, loads):
    # Randomly choose individuals from the population to participate in the tournament
    tournament_candidates = random.sample(population, tournament_size)

    # Evaluate the fitness of each candidate in the tournament
    tournament_fitness = [total_distance_with_vehicle_penalty(candidate, loads, penalty_weight) for candidate in tournament_candidates]

    # Choose the individual with the best fitness (lowest total distance with penalties) as the parent
    parent = tournament_candidates[tournament_fitness.index(min(tournament_fitness))]

    return parent

def mutate(Routes, loads, capacity, IterNum = 100):
    BestRoutes = Routes.copy()
    for _ in range(IterNum):
        
        SelectedRouteIdx = np.random.randint(0, len(Routes))
        SelectedRoute = BestRoutes[SelectedRouteIdx].copy()
        if len(SelectedRoute) > 2:

            i, j = np.random.randint(1, len(SelectedRoute) - 1, size=2)

            if j < i:
                i, j = j, i

            NewRoutes = BestRoutes.copy()
            NewRoute = SelectedRoute.copy()
            NewRoute[i:j] = NewRoute[j-1:i-1:-1]

            NewRoutes = BestRoutes.copy()
            NewRoutes[SelectedRouteIdx] = NewRoute.copy()

            NewTotalDistanceForRoute = RouteDistance(NewRoute, loads)
            
            if NewTotalDistanceForRoute < capacity:
                
                if NewTotalDistanceForRoute < RouteDistance(BestRoutes[SelectedRouteIdx], 
                                                            loads):
                    BestRoutes = NewRoutes.copy()

    return BestRoutes


def genetic_algorithm(loads, 
                      vehicle_capacity,
                      penalty_weight, 
                      population_size, 
                      num_generations, 
                      tournament_size, 
                      crossover_rate, 
                      mutation_rate):
    
    population = generate_initial_population(population_size, 
                                            loads, 
                                            vehicle_capacity)
    # print(population)
    for pop in population:
         for rt in pop:
            if RouteDistance(rt, loads) > vehicle_capacity:
                print(rt,'error when pop')
                break
    for generation in range(num_generations):

        parents = [select_parents(population, tournament_size, loads) for _ in range(population_size)]


        # Mutation
        for i in range(len(parents)):
            if random.random() < mutation_rate:
                offspring[i] = mutate(offspring[i], loads, vehicle_capacity)
        for off in offspring:
            for rt in off:
                if RouteDistance(rt,loads) > vehicle_capacity:
                    print('error when mutt')
        # Select survivors for the next generation
        population = [select_parents(offspring, tournament_size, loads) for _ in range(population_size)]
        
        best_solution = min(population, key=lambda ind: total_distance_with_vehicle_penalty(ind, loads, penalty_weight))
        

    return best_solution



path = str(sys.argv[1])

Loads = loadProblemFromFile(path)
ld = [(Loads[i].pickup,Loads[i].dropoff) for i in range(0,len(Loads))]
capacity = 12*60


penalty_weight = 500
population_size = 20
num_generations = 10
tournament_size = 3
crossover_rate = 0.8
mutation_rate = 0.1

rt = genetic_algorithm(ld, 
                      capacity,
                      penalty_weight, 
                      population_size, 
                      num_generations, 
                      tournament_size, 
                      crossover_rate, 
                      mutation_rate)

for i in range(0,len(rt)):
    for j in range(0,len(rt[i])):
        rt[i][j] += 1
sys.stdout.write(str(rt).replace("], [","]\n[")[1:-1])