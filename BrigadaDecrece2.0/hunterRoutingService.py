# -*- coding: utf-8 -*-
"""
Routing Sin Módulos
TUL - Andrés F Romero
"""

# Importación de librerías

import math
import json
import sys

class RoutingHunterService:

    def loadData(self, file: str):
        with open(file, 'r') as fp:
            data = json.load(fp)
        return data
    
    def encoder(self, packages: dict):
        # Encoder reduce ram memory changing Strings to int32 integer
        C2Id = {} # Encoder dictionary
        count = 1 # Nodes starts at 1 because 0 is the origin
        keyList = list(packages.keys()) # List of packages
        
        for i in keyList:
            packages[count] = packages.pop(i)
            C2Id[count] = i
            count += 1
            
        return C2Id

    def sortFleet(self, fleet: dict) -> list:
        # Incremental cost heterogeneous fleet
        listFleet = [fleet[x] for x in fleet]
        return sorted(listFleet, key=lambda d: d['fixedCost'])

    def linearDistance(self, a: list, b: list, vel: float = 12) -> tuple:
        # Taking into account the world radius
        R = 6373.0

        lat1 = math.radians(a[0])
        lon1 = math.radians(a[1])
        lat2 = math.radians(b[0])
        lon2 = math.radians(b[1])

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        distance = R * c  # Distance in km
        time = distance / vel * 60  # time in minutes

        return int(distance), int(time)

    def timeMatrix(self, packages: dict, vel: float = 10) -> dict:
        # Saved in memory a list of [distance[km], time[min]]
        result = {}

        for i in packages:
            for j in packages:
                a, b = self.linearDistance(packages[i]['coordinates'], packages[j]['coordinates'])
                result[(i, j)] = [a, b]  # distancia Km, Tiempo min

        return result  # distance in Km, Time in min

    def addOrigin(self, tM: dict, packages: dict, warehouse: dict):
        # Origin is always node 0
        packages[0] = warehouse
        for i in packages:
            if i != 0:
                # All nodes (packages) are destination
                a = warehouse["coordinates"]
                b = packages[i]["coordinates"]

                tM[(0, i)] = self.linearDistance(a, b)
                tM[(i, 0)] = tM[(0, i)]

    def tsp(self, tM: dict, packages: dict, ini: int = 0) -> list:
        # Sets initialization
        visited = set([ini])
        unvisited = set([i for i in packages if i != 0])

        # Route initialization
        route = [ini]

        # Algorithm complexity O(logN); N: nodes (packages)
        while len(unvisited) > 0:
            # Solution initialization
            nearest = float("inf")
            nextnode = "-1"

            for j in unvisited:
                test = tM[route[-1], j][1]
                if test < nearest:
                    nearest = test
                    nextnode = j

            route.append(nextnode)
            visited.add(nextnode)
            unvisited.discard(nextnode)
        
        return route

    def simpleStats(self, packages: dict, tM:dict, route: list) -> float:
        # Results Init
        load = 0
        expectedTime = 0
        Exp1 = 0
        Exp2 = 0
        n= 0
        prev = 0
        for i in route:
            load += 1 # Load equal to number of visits
            expectedTime += tM[prev,i][1]
            # Mean Distance
            for j in route:
                if i != j:
                    dif = max(0,(tM[i,j][0]-1))
                    Exp1 += dif**2
                    Exp2 += dif
                    n += 1
            prev = i
        try:
            meanDistance = math.sqrt((Exp1-(Exp2**2)/n)/(n-1))
        except:
            meanDistance = 0
            
        return load, expectedTime, meanDistance

    def routeEvaluation(self, packages:dict, tM:dict, sortedFleet: list, route:list, costRef:float, summary: bool)-> dict:

        # Se inicializa el resultado
        posible = False
        cost = 9999999
        
        fxC = 0.17 # Fixed Cost Weight
        vC = 0.73  # Variable cost per hour
        mdC = 0.10 # Mean distance weight
        
        # Simple route stats
        loadR, expTime, meanDistance = self.simpleStats(packages, tM, route)
        
        for truck in sortedFleet:
            # Simple stats validator in the actual truck
            if (loadR <= truck['weightCapacity']):
                posible = True # In this point all restrictions pass
                cost = truck['fixedCost']*fxC + costRef*vC*expTime/120
                
                if summary:
                    return {'c': int(cost), 't': truck['name'] }
                else:
                    return {'expectedCost': int(cost), 'expectedTime': int(expTime), 'truck': truck['name'],'load': int(loadR)}
                    
            if (truck == sortedFleet[-1] and posible == False): 
                return False
            else:
                continue
        return False

    def preSolver(self, packages:dict, tM:dict, sortedFleet: list, costRef:float, summary: bool):
        
        excludedPackages = []
        for i in packages:
            if i != 0:
                route = [i]
                posible = self.routeEvaluation(packages, tM, sortedFleet, route, costRef, summary)
                if posible == False:
                    excludedPackages.append(i)
        for i in excludedPackages:
            packages.pop(i)
        
        return excludedPackages                

    def splitAlgorithm(self, packages: dict, tM: dict, sortedFleet: list, route: list, costRef:float,
                       evaluatedRoutes: dict) -> list:
        arcs = []
        n = len(route)

        # All nodes can be the origin of a subtour
        for i in range(n - 1):
            # Consecutive nodes
            for j in range(i + 1, n):

                testRoute = route[i + 1:j + 1]
                
                # First check the poolRoutes dictionary in order to save time
                if (str(testRoute) in evaluatedRoutes):
                        subRoute = evaluatedRoutes[str(testRoute)]
                else:
                    subRoute = self.routeEvaluation(packages, tM, sortedFleet, testRoute, costRef, True)
                    evaluatedRoutes[str(testRoute)] = subRoute

                # Only posible arcs are created
                if type(subRoute) == bool:
                    break
                else:
                    arcs.append([route[i], route[j], subRoute['c']])
        return arcs

    def bellmanFord(self, arcs: list, route: list, origin: int = 0 ) -> dict:

        solution = {}
        # BF initialization
        solution = {i: [origin, float('inf')] for i in route if i != origin}
        solution[origin] = [origin, 0]

        # One iteration O(n)
        for arc in arcs:
            dv = solution[arc[1]][1]
            du = solution[arc[0]][1]
            edge = arc[2]

            test = du + edge # Edge is only the integer cost
            if dv > test:
                solution[arc[1]] = [arc[0], test]

        return solution  # Predecessor, Cost

    def simpleCost(self, tM: dict, bigRoute: list) -> float:
        return sum([tM[(bigRoute[i], bigRoute[i + 1])][1] for i in range(len(bigRoute) - 1)])

    def _2optSwap(self, bigRoute: list, i: int, k: int) -> list:
        firstPart = bigRoute[0:i]
        secondPart = bigRoute[i:k + 1]
        thirdPart = bigRoute[k + 1:len(bigRoute)]
        return firstPart + secondPart[::-1] + thirdPart

    def _2opt(self, packages: dict, tM: dict, sortedFleet: dict, evaluatedRoutes: dict,
              bigRoute: list, costRef, maxIter: int = 5000, depht: int = 15) -> list:

        n = len(bigRoute) # Nodes

        # Solution Initialization
        route = bigRoute
        bestRoute = bigRoute
        bestArcs = self.splitAlgorithm(packages, tM, sortedFleet, bestRoute, costRef, evaluatedRoutes)
        bestSolution = self.bellmanFord(bestArcs, bestRoute)
        bestCost = bestSolution[bestRoute[-1]][1]

        iterC = 1 # Iteration counter

        improvement = True
        while improvement == True:
            improvement = False

            for i in range(1, n - 1):
                if iterC > maxIter:
                    improvement = False
                    break

                for j in range(i + 1, min(n, i + 1 + depht)):
                    if iterC > maxIter:
                        improvement = False
                        break

                    # Swap and cost
                    newRoute = self._2optSwap(route, i, j)
                    newArcs = self.splitAlgorithm(packages, tM, sortedFleet, newRoute, costRef, evaluatedRoutes)
                    newSolution = self.bellmanFord(newArcs, newRoute)
                    newCost = newSolution[newRoute[-1]][1]
                    
                    # NewCost Evaluation
                    if newCost < bestCost:
                        bestRoute = newRoute
                        bestArcs = newArcs
                        bestSolution = newSolution
                        bestCost = newCost
                        improvement = True

                iterC += 1
                route = bestRoute
        
        last = bestRoute[-1]
        prev = -1
        rta = []
        
        while prev !=0:
            prev = bestSolution[last][0]
            for arc in bestArcs:
                origin = arc[0]
                destination = arc[1]
                
                if (origin == prev and destination == last):
                    orIndex = bestRoute.index(origin)
                    dsIndex = bestRoute.index(destination)
                    rta.append(str(route[orIndex+1:dsIndex+1]))
            last = prev
        return rta

    def simple2opt(self, tM: dict, bigRoute: list) -> list:

        n = len(bigRoute) # Nodes

        # Solution Initialization
        route = bigRoute
        bestRoute = bigRoute
        bestCost = self.simpleCost(tM, bestRoute)

        improvement = True
        while improvement == True:
            improvement = False
            for i in range(1, n - 1):
                for j in range(i + 1, n):
                    newRoute = self._2optSwap(route, i, j)
                    newCost = self.simpleCost(tM, newRoute)

                    if newCost < bestCost:
                        bestRoute = newRoute
                        bestCost = newCost
                        improvement = True

                route = bestRoute

        return bestRoute
    
    def printResults(self, excludedPackages: list, usedRoutes:dict, poolRoutes:dict, C2Id:dict, 
                     packages:dict, tM:dict, sortedFleet:list, costRef:int, 
                     origin:int = 0 ) -> dict:
        
        solution = []
        for i in usedRoutes:
            route = i['r']
            # result = self.routeEvaluation(packages, tM, sortedFleet, route, costRef, False)
            route = [ {'id': self.packages[j]['id'], 'type': self.packages[j]['type'] } for j in route ]
            solution.append(route)
        
        return solution

    # =============================================================================
    # Global Variables
    # =============================================================================

    fleet = {
            'Hunter': {
                'name': 'Hunter',
                'weightCapacity': 30,
                'fixedCost': 100
            }
        }
    warehouse = {}
    packages = {}
    id = ""

    # =============================================================================
    # MAIN
    # =============================================================================
    def hunterRouting(self, data):
        # Rewrite global variables
        self.packages = { i['id']:i for i in data['ironmongeries'] }
        self.id = data['id']

        # Parameters preparation
        C2Id = self.encoder(self.packages)
        sortedFleet = self.sortFleet(self.fleet)
        costRef = int(sortedFleet[-1]['fixedCost'])
        
        for i in self.packages:
            self.packages[i]['coordinates'] = [float(self.packages[i]['coordinates'][0]), float(self.packages[i]['coordinates'][1])]
        try:
            tM = self.timeMatrix(self.packages)
        except:
            sys.exit('No se pudo obtener una matriz provisional de distancias')

        # Adding origin
        self.warehouse = self.packages[1]
        self.warehouse['maxIter'] = len(self.packages)*6
        self.warehouse['depht'] = 6

        try:
            self.addOrigin(tM, self.packages, self.warehouse)
        except:
            sys.exit('No se pudo agregar el origen a la lista de paquetes')
            
        # Presolve, infactible packages are excluded from the algorithm first
        # try:
        excludedPackages = self.preSolver(self.packages, tM, sortedFleet, costRef, True)
        # except:
        #     sys.exit("Presolver Failure")

        # =============================================================================
        # Split Algorithm - Metaheuristic
        # =============================================================================
        try:
            bigRoute = self.tsp(tM, self.packages)
        except:
            sys.exit('Falló la solución TSP del problema inicial')

        try:
            bigRoute = self.simple2opt(tM, bigRoute)
        except:
            sys.exit('No se pudo mejorar la solución inicial con 2optSimple')

        # =============================================================================
        # Iterative upgrade of the solution
        # =============================================================================

        evaluatedRoutes = {} # Pool routes
        
        # Iterative Procedure. This solution is the warm start
        usedRoutes = self._2opt(self.packages, tM, sortedFleet, evaluatedRoutes,
                        bigRoute, costRef, self.warehouse['maxIter'], self.warehouse['depht'])
        usedRoutes = [{'r': json.loads(i), 'c': evaluatedRoutes[i]['c'], 't': evaluatedRoutes[i]['t']} for i in usedRoutes]
        return { 
                'id': self.id,
                'routes':self.printResults(excludedPackages, usedRoutes, evaluatedRoutes, C2Id, self.packages, tM, sortedFleet, costRef) 
                }