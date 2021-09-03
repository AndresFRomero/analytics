# -*- coding: utf-8 -*-
"""
Routing Sin Módulos
TUL - Andrés F Romero
"""

# Importación de librerías

import math
import json
import sys
import requests
import time


class RoutingTruckService:

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
            
    def timeConversion(self, strTime: str):
        h, m, s = strTime.split(":")
        h = int(h)
        m = int(m)
        s = int(s)
        
        return int(h * 60 + m + s / 6)
    
    def timeString(self, departure: float) -> str:
        # Reverse time conversion from int to str
        h = str(math.floor(departure / 60))
        m = str(math.ceil(departure - float(h) * 60))
        
        return h.zfill(2) + ":" + m.zfill(2) + ":00"

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

    def realDistance(self, a: list, b: list) -> tuple:
        # HereMaps Routing API consumer
        url = 'https://router.hereapi.com/v8/routes'

        departureTime = time.gmtime(time.time() + 24 * 60 * 60)
        departureTime = str(departureTime.tm_year).zfill(4) + '-' + str(departureTime.tm_mon).zfill(2) + \
                        '-' + str(departureTime.tm_mday).zfill(2) + 'T08:00:00'

        payload = {
            'transportMode': 'truck',
            'origin': str(a[0]) + ',' + str(a[1]),
            'destination': str(b[0]) + ',' + str(b[1]),
            'routingMode': 'fast',
            'return': 'summary',
            'departureTime': departureTime,
            'apiKey': '8337lfqwsU7Oaz_3cuo2UUaxrG7y4Hq7tk4vXUzMv5k'
        }

        response = requests.get(url, params=payload)
        response = response.json()

        # Siempre se retorna distancia en kilómetros y tiempo en minutos
        return [int(response['routes'][0]['sections'][0]['summary']['length'] / 1000),
                int(response['routes'][0]['sections'][0]['summary']['duration'] / 60)]

    def timeMatrix(self, packages: dict, vel: float = 10) -> dict:
        # Saved in memory a list of [distance[km], time[min]]
        result = {}

        for i in packages:
            for j in packages:
                a, b = self.linearDistance(packages[i]['coordinates'], packages[j]['coordinates'])
                result[(i, j)] = [a, b]  # distancia Km, Tiempo min

        return result  # distance in Km, Time in min

    def usageDistance(self, tM: dict, c: float = 1) -> float:
        # Utilization percentage
        n = math.sqrt(len(tM))

        distanceList = [tM[x][0] for x in tM]
        distanceList = sorted(distanceList)
        usage = min(c * n + n, len(distanceList))
        try:
            return distanceList[int(usage) - 1]
        except:
            return 1

    def topConection(self, tM: dict, packages: dict):
        # Top nodes conection by diferent locations
        maxConection = {}

        for i in packages:
            distances = list(set([round(tM[j][0], 2) for j in tM if j[0] == i]))
            distances.sort()

            maxNeibor = min(10, len(distances))
            maxConection[i] = distances[maxNeibor - 1]

        return maxConection

    def tmUpdate(self, tM: dict, packages: dict, maxD: float):
        # Linear distance matrix recalculated
        linearTM = tM.copy()
        maxConection = self.topConection(linearTM, packages)

        for i in tM:
            origin = packages[i[0]]['coordinates']
            destination = packages[i[1]]['coordinates']

            # Usage distance and topConection validator
            if (tM[i][0] <= maxD or tM[i][0] < maxConection[i[0]]) and tM[i][0] != 0:
                try:
                    rD = self.linearDistance(origin, destination)
                    tM[i] = [max(tM[i][0], rD[0]), max(tM[i][1], rD[1])]
                # If error occurs, linear distance is kept
                except:
                    pass

    def addOrigin(self, tM: dict, packages: dict, warehouse: dict):
        # Origin is always node 0
        packages[0] = warehouse
        
        for i in packages:
            if i != 0:
                # All nodes (packages) are destination
                a = warehouse["coordinates"]
                b = packages[i]["coordinates"]
                try:
                    tM[(0, i)] = self.realDistance(a, b)
                    tM[(i, 0)] = tM[(0, i)]
                except:
                    tM[(0, i)] = self.linearDistance(a, b)
                    tM[(i, 0)] = tM[(0, i)]

    def serviceTime(self, packages: dict):
        for i in packages:
            # Basic time service ecuation
            x = (packages[i]['weight']-3000)/1000
            packages[i]['service'] = 1/(1 + math.exp(-1.4*(x + 0.25)))*180 + 15

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
        volume = 0
        length = 0
        Exp1 = 0
        Exp2 = 0
        n= 0
        
        for i in route:
            load += packages[i]['weight']
            volume += packages[i]['volume']
            length = max(length, packages[i]['length'])
            
            # Mean Distance
            for j in route:
                if i != j:
                    dif = max(0,(tM[i,j][0]-1))
                    Exp1 += dif**2
                    Exp2 += dif
                    n += 1
        try:
            meanDistance = math.sqrt((Exp1-(Exp2**2)/n)/(n-1))
        except:
            meanDistance = 0
            
        return load, volume, length, meanDistance

    def expectedTime(self, packages: dict, tM: dict, route: list, departure: float = 0,
                     tax: float = 84000) -> tuple:
        # Params Initialization
        prev = 0
        clock = departure
        penalties = 0
        waits = []

        for i in route:
            # Arrival time to the point
            clock += tM[(prev, i)][1]
            testS = packages[i]['start']
            testE = packages[i]['end']

            # Decision whether to deliver or wait
            if clock < testS:
                waits.append(testS - clock)
                clock = testS

            elif clock > testE:
                penalties += tax

            # Service time addition to route
            if tM[(prev, i)][1] >= 2:
                # If time is greater than 2mins, then is the same point
                clock += packages[i]['service']
            else:
                # Add only variable service time
                clock += packages[i]['service'] - 15

            # Previous package actualization
            prev = i

        return clock - departure, penalties, waits

    def finalExpectedTime(self, packages: dict, tM: dict, route: list, tax: float):

        # Departure at 00:00:00
        iniTime, iniPenalties, iniWaits = self.expectedTime(packages, tM, route, 0, tax)

        # Pushing waits.
        push = 0
        departure = 0
        
        for i in iniWaits:
            push += i
            testTime, testPenalties, testWaits = self.expectedTime(packages, tM, route, push, tax)

            if testTime < iniTime and testPenalties <= iniPenalties:
                # If a better solution is achived, params are refreshed
                iniTime = testTime
                iniPenalties = testPenalties
                departure = push

        return iniTime, iniPenalties, self.timeString(departure)

    def routeEvaluation(self, packages:dict, tM:dict, sortedFleet: list, route:list, costRef:float, summary: bool)-> dict:

        # Se inicializa el resultado
        posible = False
        cost = 9999999
        
        fxC = 0.17 # Fixed Cost Weight
        vC = 0.52  # Variable cost per hour
        pC = 0.12  # Penalties cost weight
        eC = 0.12  # Excess cost weight
        mdC = 0.07 # Mean distance weight
        
        # Simple route stats
        loadR, volumeR, lengthR, meanDistance = self.simpleStats(packages, tM, route)
        
        for truck in sortedFleet:
            # Weight excess restriction fixed in 10%
            excessR = max(0,  loadR - truck['weightCapacity'])
            maxExcess = truck['weightCapacity']*0.1
            excessPercentage = excessR/maxExcess
            
            # Simple stats validator in the actual truck
            if (excessR <= maxExcess 
                and lengthR <= truck['lengthCapacity']
                and volumeR <= truck['volumeCapacity']):
                
                # Time validator
                expTime, penalties, departure = self.finalExpectedTime(packages, tM, route, costRef*0.1)
                
                if (expTime <= truck['timeCapacity']):   
                    posible = True # In this point all restrictions pass
                    cost = truck['fixedCost']*fxC + costRef*vC*expTime/60 + penalties*pC + excessPercentage*costRef*eC + costRef*meanDistance*mdC
                    
                    if summary:
                        return int(cost)
                    else:
                        return {'posible': posible, 'expectedCost': int(cost), 'expectedTime': int(expTime), 'truck': truck['name'],'load': int(loadR)}
                    
            if (truck == sortedFleet[-1] and posible == False and len(route) == 1): 
                if summary:
                    return int(cost)
                else:
                    # Time validator
                    expTime, penalties, departure = self.finalExpectedTime(packages, tM, route, costRef*pC)
                    return {'posible': True, 'expectedCost': int(cost), 'expectedTime': int(expTime), 'truck': 'Infac', 'load': int(loadR)}
            else:
                continue
        
        return False

    """ Función de split optimizada.
        La idea es que la lista de camiones esté ordenada por preferencia de uso
    """

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
                    arcs.append([route[i], route[j], subRoute])
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

        return [bestRoute, bestArcs, bestSolution, bestCost]

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

    def printResults(self, best2opt: list, C2Id: dict, packages: dict, tM: dict, sortedFleet: list, costRef: int, origin: int = 0 ) -> dict:

        route = best2opt[0]
        arcs = best2opt[1]
        solution = best2opt[2]

        last = route[-1]
        prev = "-1"

        rta = []
        while prev != 0:
            prev = solution[last][0]
            for arc in arcs:
                origin = arc[0]
                destination = arc[1]
                
                orIndex = route.index(origin)
                dsIndex = route.index(destination)
                
                subRoute = arc[2]
                packs = route[orIndex+1:dsIndex+1] # Packages in route
                subRoute = self.routeEvaluation(packages, tM, sortedFleet, packs, costRef, False)
                packs = [ C2Id[i] for i in packs ] #Reverse Encode
                subRoute['packages'] = packs

                if (origin == prev and destination == last):
                    rta.append(subRoute)

            last = prev

        return rta

    # =============================================================================
    # Fleet of current warehouses
    # =============================================================================

    fullFleet = {
        'bodega Fontibon': {
            'Carry': {
                'name': 'Carry',
                'weightCapacity': 800,
                'volumeCapacity': 3.8,
                'timeCapacity': 360,
                'lengthCapacity': 2,
                'fixedCost': 120000
            },
            'NHR': {
                'name': 'NHR',
                'weightCapacity': 2200,
                'volumeCapacity': 20,
                'timeCapacity': 360,
                'lengthCapacity': 4.5,
                'fixedCost': 280000
            },
            'Turbo': {
                'name': 'Turbo',
                'weightCapacity': 5000,
                'volumeCapacity': 22,
                'timeCapacity': 360,
                'lengthCapacity': 6.3,
                'fixedCost': 320000
            },
            'Turbo_Extra_Dim': {
                'name': 'Turbo_Extra_Dim',
                'weightCapacity': 7000,
                'volumeCapacity': 22,
                'timeCapacity': 360,
                'lengthCapacity': 6.3,
                'fixedCost': 360000
            },
            'Sencillo': {
                'name': 'Sencillo',
                'weightCapacity': 10000,
                'volumeCapacity': 22,
                'timeCapacity': 360,
                'lengthCapacity': 6.3,
                'fixedCost': 420000
            }
        },
        'Bodega Bucaramanga': {
            'Turbo': {
                'name': 'Turbo',
                'weightCapacity': 5000,
                'volumeCapacity': 22,
                'timeCapacity': 360,
                'lengthCapacity': 6.3,
                'fixedCost': 320000
            },
            'Turbo_Extra_Dim': {
                'name': 'Turbo_Extra_Dim',
                'weightCapacity': 7000,
                'volumeCapacity': 22,
                'timeCapacity': 360,
                'lengthCapacity': 6.3,
                'fixedCost': 360000
            },
            'Sencillo': {
                'name': 'Sencillo',
                'weightCapacity': 10000,
                'volumeCapacity': 22,
                'timeCapacity': 360,
                'lengthCapacity': 6.3,
                'fixedCost': 420000
            }
        },
        'Bodega Cali': {
            'Turbo': {
                'name': 'Turbo',
                'weightCapacity': 5000,
                'volumeCapacity': 22,
                'timeCapacity': 360,
                'lengthCapacity': 6.3,
                'fixedCost': 320000
            },
            'Turbo_Extra_Dim': {
                'name': 'Turbo_Extra_Dim',
                'weightCapacity': 7000,
                'volumeCapacity': 22,
                'timeCapacity': 360,
                'lengthCapacity': 6.3,
                'fixedCost': 360000
            },
            'Sencillo': {
                'name': 'Sencillo',
                'weightCapacity': 10000,
                'volumeCapacity': 22,
                'timeCapacity': 360,
                'lengthCapacity': 6.3,
                'fixedCost': 420000
            }
        },
        'Quito': {
            'Turbo': {
                'name': 'Turbo',
                'weightCapacity': 5000,
                'volumeCapacity': 32000,
                'lengthCapacity': 6,
                'timeCapacity': 360,
                'pointCost': 120,
                'fixedCost': 120,
                'variableCost': 0
            },
            'TurboExtraDim': {
                'name': 'TurboExtraDim',
                'weightCapacity': 7000,
                'volumeCapacity': 32000,
                'lengthCapacity': 12,
                'timeCapacity': 360,
                'pointCost': 135,
                'fixedCost': 135,
                'variableCost': 0
            },
            'Sencillo': {
                'name': 'Sencillo',
                'weightCapacity': 10000,
                'volumeCapacity': 32000,
                'lengthCapacity': 12,
                'timeCapacity': 360,
                'pointCost': 150,
                'fixedCost': 150,
                'variableCost': 0
            }
        },
        'Guayaquil': {
            'Turbo': {
                'name': 'Turbo',
                'weightCapacity': 5000,
                'volumeCapacity': 32000,
                'lengthCapacity': 6,
                'timeCapacity': 360,
                'pointCost': 120,
                'fixedCost': 120,
                'variableCost': 0
            },
            'TurboExtraDim': {
                'name': 'TurboExtraDim',
                'weightCapacity': 7000,
                'volumeCapacity': 32000,
                'lengthCapacity': 12,
                'timeCapacity': 360,
                'pointCost': 135,
                'fixedCost': 135,
                'variableCost': 0
            },
            'Sencillo': {
                'name': 'Sencillo',
                'weightCapacity': 10000,
                'volumeCapacity': 32000,
                'lengthCapacity': 12,
                'timeCapacity': 360,
                'pointCost': 150,
                'fixedCost': 150,
                'variableCost': 0
            }
        },
        'Bodega Medellin': {
            'Carry': {
                'name': 'Carry',
                'weightCapacity': 800,
                'volumeCapacity': 3.8,
                'timeCapacity': 360,
                'lengthCapacity': 2,
                'fixedCost': 120000
            },
            'Turbo': {
                'name': 'Turbo',
                'weightCapacity': 5000,
                'volumeCapacity': 22,
                'timeCapacity': 360,
                'lengthCapacity': 6.3,
                'fixedCost': 320000
            },
            'Turbo_Extra_Dim': {
                'name': 'Turbo_Extra_Dim',
                'weightCapacity': 7000,
                'volumeCapacity': 22,
                'timeCapacity': 360,
                'lengthCapacity': 6.3,
                'fixedCost': 360000
            },
            'Sencillo': {
                'name': 'Sencillo',
                'weightCapacity': 10000,
                'volumeCapacity': 22,
                'timeCapacity': 360,
                'lengthCapacity': 6.3,
                'fixedCost': 420000
            }
        },
        'Bodega Barranquilla': {
            'Carry': {
                'name': 'Carry',
                'weightCapacity': 800,
                'volumeCapacity': 3.8,
                'timeCapacity': 360,
                'lengthCapacity': 2,
                'fixedCost': 120000
            },
            'Turbo': {
                'name': 'Turbo',
                'weightCapacity': 5000,
                'volumeCapacity': 22,
                'timeCapacity': 360,
                'lengthCapacity': 6.3,
                'fixedCost': 320000
            },
            'Turbo_Extra_Dim': {
                'name': 'Turbo_Extra_Dim',
                'weightCapacity': 7000,
                'volumeCapacity': 22,
                'timeCapacity': 360,
                'lengthCapacity': 6.3,
                'fixedCost': 360000
            },
            'Sencillo': {
                'name': 'Sencillo',
                'weightCapacity': 10000,
                'volumeCapacity': 22,
                'timeCapacity': 360,
                'lengthCapacity': 6.3,
                'fixedCost': 420000
            }
        },
        'Tlalnepantla': {
            'TurboExtraDim': {
                'name': 'TurboExtraDim',
                'weightCapacity': 7000,
                'volumeCapacity': 32000,
                'lengthCapacity': 6,
                'timeCapacity': 360,
                'pointCost': 2050,
                'fixedCost': 1228,
                'variableCost': 128
            },
            'Sencillo': {
                'name': 'Sencillo',
                'weightCapacity': 10000,
                'volumeCapacity': 32000,
                'lengthCapacity': 9,
                'timeCapacity': 360,
                'pointCost': 2340,
                'fixedCost': 1400,
                'variableCost': 146
            }
        }
    }

    # =============================================================================
    # MAIN
    # =============================================================================
    def initTruckRouting(self, data):

        # Encoder
        C2Id = self.encoder(data['packages'])
        
        # Time Service actualization
        self.serviceTime(data['packages'])
        
        # Timeout and Ram
        data['warehouse']['maxIter'] = len(data['packages'])*15
        data['warehouse']['depht'] = 6
        
        for i in data['packages']:
            data['packages'][i]['start'] = self.timeConversion(data['packages'][i]['start'])
            data['packages'][i]['end'] = self.timeConversion(data['packages'][i]['end'])
            data['packages'][i]['coordinates'] = [float(data['packages'][i]['coordinates'][0]),
                                                  float(data['packages'][i]['coordinates'][1])]
        try:
            tM = self.timeMatrix(data['packages'])
        except:
            sys.exit('No se pudo obtener una matriz provisional de distancias')

        # Usage Distance Calculation
        try:
            maxD = self.usageDistance(tM)
        except:
            sys.exit('No se pudo obtener una distancia máxima de utilización')

        # Real Distance update
        try:
            self.tmUpdate(tM, data['packages'], maxD)
        except:
            sys.exit('No se pudo actualizar la matriz de DT a T usando Api')

        # Adding origin
        try:
            self.addOrigin(tM, data['packages'], data['warehouse'])
        except:
            sys.exit('No se pudo agregar el origen a la lista de paquetes')

        # =============================================================================
        # Split Algorithm - Metaheuristic
        # =============================================================================
        try:
            bigRoute = self.tsp(tM, data['packages'])
        except:
            sys.exit('Falló la solución TSP del problema inicial')

        try:
            if len(data['fleet']) == 0:
                try:
                    sortedFleet = self.sortFleet(self.fullFleet[data['warehouse']['name']])
                except:
                    sortedFleet = self.sortFleet(self.fullFleet['bodega Fontibon'])
            else:
                sortedFleet = self.sortFleet(data['fleet'])
        except:
            sys.exit('No se pudo obtener la flota asociada a la bodega')
            
        costRef = int(sortedFleet[-1]['fixedCost'])

        try:
            bigRoute = self.simple2opt(tM, bigRoute)
        except:
            sys.exit('No se pudo mejorar la solución inicial con 2optSimple')

        # =============================================================================
        # Iterative upgrade of the solution
        # =============================================================================

        evaluatedRoutes = {} # Pool routes

        # Iterative Procedure
        bestRoute, bestArcs, bestSolution, bestCost = self._2opt(data['packages'], tM, sortedFleet, evaluatedRoutes,
                                            bigRoute, costRef, data['warehouse']['maxIter'], data['warehouse']['depht'])
        
        return self.printResults([bestRoute, bestArcs, bestSolution, bestCost], C2Id, data['packages'], tM, sortedFleet, costRef)