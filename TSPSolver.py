#!/usr/bin/python3
import copy

from which_pyqt import PYQT_VER

if PYQT_VER == 'PYQT5':
    from PyQt5.QtCore import QLineF, QPointF
else:
    raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))

import time
import numpy as np
from TSPClasses import *
import heapq
import itertools

class HeapQueue:
    def __init__(self):
        self.nodes = []
        # Probably need a dictionary for the indices for easy adding/removal
        self.array_indices = {}
        self.max_queue_size = 0

    def delete_min(self):

        # Parent index is 0
        parent_index = 0

        # Save node at top of heap
        node_to_remove = self.nodes[parent_index]

        # Get node at end of heap
        heap_end_node = self.nodes[self.size() - 1]

        # Remove node from end of heap
        self.nodes.remove(heap_end_node)

        # If node at top was final node,
        if self.size() == 0:
            del self.array_indices[node_to_remove]
            return node_to_remove

        # Set top node to final node in heap
        self.nodes[parent_index] = heap_end_node

        # Set removed node index to -1
        del self.array_indices[node_to_remove]

        # Set end nodes index to 0/parent node since it is at top now
        self.array_indices[heap_end_node] = parent_index

        # percolate node downwards until it stops
        while True:

            # Index of left and right children
            left_child_index = (parent_index * 2) + 1
            right_child_index = (parent_index * 2) + 2

            # Get parent node and lowerbound
            parent_node = self.nodes[parent_index]
            parent_lower_bound = parent_node.lowerbound

            # Make priority key not just lower-bound
            parent_priority = parent_lower_bound / len(parent_node.path)

            # If left child exists in array
            if left_child_index < self.size():
                left_child_node = self.nodes[left_child_index]
                left_child_lower_bound = left_child_node.lowerbound
                left_child_priority = left_child_lower_bound / len(left_child_node.path)
            # If there is no left child (and by extension no right child)
            else:
                left_child_priority = np.inf
                left_child_node = None

            # If right child exists in array
            if right_child_index < self.size():
                right_child_node = self.nodes[right_child_index]
                right_child_lower_bound = right_child_node.lowerbound
                right_child_priority = right_child_lower_bound / len(right_child_node.path)
            else:
                right_child_priority = np.inf
                right_child_node = None

            parent_swapped = False

            # If left child lowerbound is less than right, use it, tie break left
            if left_child_priority <= right_child_priority:
                # If left child is less than parent, update, else do nothing
                if left_child_priority < parent_priority:
                    self.nodes[parent_index] = left_child_node
                    self.nodes[left_child_index] = parent_node

                    self.array_indices[parent_node] = left_child_index
                    self.array_indices[left_child_node] = parent_index

                    parent_index = left_child_index
                    parent_swapped = True
            else:  # If right child lowerbound is less than parent, check for swap
                if right_child_priority < parent_priority:
                    self.nodes[parent_index] = right_child_node
                    self.nodes[right_child_index] = parent_node

                    self.array_indices[parent_node] = right_child_index
                    self.array_indices[right_child_node] = parent_index

                    parent_index = right_child_index
                    parent_swapped = True

            if not parent_swapped:
                break

        return node_to_remove

    # Place new or updated node into correct spot on heap queue
    def decrease_key(self, node):

        if node not in self.array_indices:
            return

        child_index = self.array_indices[node]

        # Percolate children up if parent needs to move
        while True:

            # New node reached top of heap, stop percolating
            if child_index == 0:
                break

            # Get node from array using index
            child_node = self.nodes[child_index]

            # Get parent index using child index - 1 // 2
            parent_index = (child_index - 1) // 2
            parent_node = self.nodes[parent_index]

            parent_priority = parent_node.lowerbound / len(parent_node.path)

            child_priority = child_node.lowerbound / len(child_node.path)

            if child_priority < parent_priority:

                self.nodes[child_index] = parent_node
                self.nodes[parent_index] = child_node

                self.array_indices[child_node] = parent_index
                self.array_indices[parent_node] = child_index

                child_index = parent_index

            else:
                break

    # Insert state into priority heap queue
    def insert(self, node):
        # Add node to array storage
        self.nodes.append(node)

        if self.size() > self.max_queue_size:
            self.max_queue_size = self.size()

        # Save array index to indices dictionary using node as key
        self.array_indices[node] = len(self.nodes) - 1

        # Check for shifting node up
        self.decrease_key(node)

    # Return number of nodes in heap queue
    def size(self):
        return len(self.nodes)


class Node:
    def __init__(self, lower_bound, parent_matrix, curr_city, curr_city_index, parent_path, parent_cities):
        self.costmatrix = copy.deepcopy(parent_matrix)
        self.lowerbound = lower_bound

        # Add get path up to this point and add new city to path
        self.path = []
        for city in parent_path:
            self.path.append(city)
        self.path.append(curr_city)

        # Get index of cities visited up to this point for use in cost matrix
        self.cities_visited = []
        for city in parent_cities:
            self.cities_visited.append(city)
        self.cities_visited.append(curr_city_index)

    def generateCostMatrix(self, cities):
        self.costmatrix = np.zeros((len(cities), len(cities)))

        for sourceindex, city in enumerate(cities):
            for destindex, secondcity in enumerate(cities):
                if city == secondcity:
                    self.costmatrix[sourceindex][destindex] = np.inf
                else:
                    self.costmatrix[sourceindex][destindex] = city.costTo(secondcity)

    # Reduce a new state and update the lower bound
    def reduceMatrix(self, cities, dest_city_index):
        # Get previous city index from cities_visited array
        source_city_index = self.cities_visited[len(self.cities_visited) - 2]

        # Get cost of selected path
        path_cost = self.costmatrix[source_city_index][dest_city_index]

        # Set path and inverse to infinity
        self.costmatrix[source_city_index][dest_city_index] = np.inf
        self.costmatrix[dest_city_index][source_city_index] = np.inf

        # Set row of source city and column of dest city to inf
        for city_index, city in enumerate(cities):
            self.costmatrix[source_city_index][city_index] = np.inf
            self.costmatrix[city_index][dest_city_index] = np.inf

        # Reduce updated cost matrix and get value of reductions
        rowminvalues = np.amin(self.costmatrix, axis=1)

        for rowindex, city in enumerate(self.costmatrix):
            # Get min value from min value by row array
            reducevalue = rowminvalues[rowindex]
            # If min value is infinity then set to 0 in array to not mess up sum of lower bound
            if reducevalue == np.inf:
                rowminvalues[rowindex] = 0

            for colindex, secondcity in enumerate(self.costmatrix):
                if self.costmatrix[rowindex][colindex] != 0 and reducevalue != np.inf:
                    self.costmatrix[rowindex][colindex] -= reducevalue

        colminvalues = np.amin(self.costmatrix, axis=0)

        for colindex, city in enumerate(self.costmatrix):
            # Get min value from min value by column array
            reducevalue = colminvalues[colindex]

            # If min value is infinity then set to 0 in array to not mess up sum of lower bound
            if reducevalue == np.inf:
                colminvalues[colindex] = 0

            for rowindex, secondcity in enumerate(self.costmatrix):

                if self.costmatrix[rowindex][colindex] != 0 and reducevalue != np.inf:
                    self.costmatrix[rowindex][colindex] -= reducevalue

        reduction_cost = np.sum(rowminvalues) + np.sum(colminvalues)

        # Get new bound by adding previous lower bound, path cost and reduction cost
        self.lowerbound += reduction_cost + path_cost

    def getlowerbound(self):
        # Reduce values in each row of array
        rowminvalues = np.amin(self.costmatrix, axis=1)

        for rowindex, city in enumerate(self.costmatrix):
            for colindex, secondcity in enumerate(self.costmatrix):
                reducevalue = rowminvalues[rowindex]
                if self.costmatrix[rowindex][colindex] != 0:
                    self.costmatrix[rowindex][colindex] -= reducevalue

        colminvalues = np.amin(self.costmatrix, axis=0)

        for colindex, city in enumerate(self.costmatrix):
            for rowindex, secondcity in enumerate(self.costmatrix):
                reducevalue = colminvalues[colindex]
                if self.costmatrix[rowindex][colindex] != 0:
                    self.costmatrix[rowindex][colindex] -= reducevalue

        self.lowerbound += np.sum(rowminvalues) + np.sum(colminvalues)

    def get_greedy_path(self, cities):
        source_index = self.cities_visited[len(self.cities_visited) - 1]

        city_row = self.costmatrix[source_index,:]

        dest_index = -1
        min_cost = np.inf

        # Get minimum value and index
        for index, city in enumerate(city_row):
            # Skip checking for any cities not visited yet
            if index in self.cities_visited:
                continue

            # Check for min value
            if city_row[index] < min_cost:
                min_cost = city_row[index]
                dest_index = index

        # No valid path from this city, solution is non existent
        if min_cost == np.inf:
            self.lowerbound = min_cost
        else:
            self.cities_visited.append(dest_index)
            self.path.append(cities[dest_index])
            self.lowerbound += min_cost

            # Check for valid path from final city to start
            if self.test(cities):
                return_path = self.costmatrix[dest_index][self.cities_visited[0]]

                # Path to starting node equals infinity, there is no valid return path, don't update solution
                self.lowerbound += return_path




    # Check if is a complete path, return lowerbound which is the final cost
    def test(self, cities):
        # Incomplete path, return infinity
        if len(self.path) != len(cities):
            return False

        # If path is complete, return lower bound
        return True


    def printCostMatrix(self):
        arrayString = "X\t"
        val = 0
        while val < len(self.costmatrix):
            arrayString += str(val) + "\t\t\t"
            val += 1

        arrayString += '\n'

        for sourceindex, city in enumerate(self.costmatrix):
            arrayString += str(sourceindex) + "\t"
            for destindex, secondcity in enumerate(self.costmatrix):
                if self.costmatrix[sourceindex][destindex] == np.inf or self.costmatrix[sourceindex][destindex] == 0:
                    arrayString += str(self.costmatrix[sourceindex][destindex]) + "\t\t\t"
                else:
                    arrayString += str(self.costmatrix[sourceindex][destindex]) + "\t\t"

            arrayString += '\n'

        print(arrayString)




class TSPSolver:
    def __init__(self, gui_view):
        self._scenario = None

    def setupWithScenario(self, scenario):
        self._scenario = scenario

    def defaultRandomTour(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        bssf = None
        start_time = time.time()
        while not foundTour and time.time() - start_time < time_allowance:
            # create a random permutation
            perm = np.random.permutation(ncities)
            route = []
            # Now build the route using the random permutation
            for i in range(ncities):
                route.append(cities[perm[i]])
            bssf = TSPSolution(route)
            count += 1
            if bssf.cost < np.inf:
                # Found a valid route
                foundTour = True
        end_time = time.time()
        results['cost'] = bssf.cost if foundTour else math.inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    def greedy(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)

        # Initialize Node to work with


        solution_count = 0

        # Initialize solution to infinity to avoid None error
        greedy_solution = None

        all_solutions_checked = False

        start_time = time.time()
        while all_solutions_checked is False:

            # TODO: Choose starting city randomly
            for starting_city_index in range(ncities):

                # Check if time expired
                if time.time() - start_time > time_allowance:
                    print("Time expired!")
                    break

                initial_state = None

                #print("Starting from city " + str(starting_city_index))
                initial_state = Node(0, [], cities[starting_city_index], starting_city_index, [], [])
                initial_state.generateCostMatrix(cities)

                # Loop until greedy solution finds answer
                while not initial_state.test(cities):

                    initial_state.get_greedy_path(cities)

                    # There is no valid solution for this matrix
                    if initial_state.lowerbound == np.inf:
                        break

                if greedy_solution is None:
                    # Check that initial solution is valid
                    if initial_state.lowerbound != np.inf:
                        greedy_solution = TSPSolution(initial_state.path)
                        solution_count += 1
                else:
                    if initial_state.lowerbound < greedy_solution.cost:
                        #print("Found a new solution!")
                        greedy_solution = TSPSolution(initial_state.path)
                        solution_count += 1

            #print("Tried all solutions")
            all_solutions_checked = True

        # Stop timing for end timer
        end_time = time.time()

        print("All done!")

        results['cost'] = greedy_solution.cost if solution_count > 0 else math.inf
        results['time'] = end_time - start_time
        results['count'] = solution_count
        results['soln'] = greedy_solution
        results['max'] = None
        results['total'] = None
        results['pruned'] = None

        return results


    def expand(self, curr_state, cities):
        T = []

        # Iterate over all cities
        for index, city in enumerate(cities):

            # Only create nodes for cities not visited yet for current state
            if index not in curr_state.cities_visited:
                # Create new state
                new_state = Node(curr_state.lowerbound, curr_state.costmatrix, city, index, curr_state.path, curr_state.cities_visited)

                # Reduce matrix of new state - TODO: Move call to init
                new_state.reduceMatrix(cities, index)

                # Append to result array
                T.append(new_state)

        return T


    def branchAndBound(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()

        initial_state = Node(0, [], cities[0], 0, [], [])

        # Generate initial state - TODO: handle this in the init function to clean up
        initial_state.generateCostMatrix(cities)
        initial_state.getlowerbound()

        S = HeapQueue()
        S.insert(initial_state)

        solution_count = 0
        states_pruned = 0
        total_states = 0

        # Initialize BSSF - TODO: Maybe use the greedy algorithm?
        greedy_result = self.greedy()
        bssf = greedy_result['soln']

        start_time = time.time()
        # While Heap Queue is not empty and time allowed has not passed
        while S.size() > 0 and time.time() - start_time < time_allowance:

            # Get node at top of heap
            P = S.delete_min()

            # Expand P into sub states, add to T
            T = []  # Re-empty array of expanded children states
            T = self.expand(P, cities)
            total_states += len(T)

            # For each state in T
            for Pi in T:
                # Reached a solution
                if Pi.test(cities):
                    if Pi.lowerbound < bssf.cost:
                        bssf = TSPSolution(Pi.path)
                        solution_count += 1
                # Not at a solution yet, continue with incrementing
                else:
                    # Lower bound of partial solution is >= BSSF, prune state
                    if Pi.lowerbound >= bssf.cost:
                        states_pruned += 1
                    # Path lower bound is < BSSF, push to heap queue and wait for next round
                    else:
                        S.insert(Pi)

        # Stop timing for end timer
        end_time = time.time()

        results['cost'] = bssf.cost if solution_count > 0 else math.inf
        results['time'] = end_time - start_time
        results['count'] = solution_count
        results['soln'] = bssf
        results['max'] = S.max_queue_size
        results['total'] = total_states
        results['pruned'] = states_pruned

        return results





    ''' <summary>
        This is the entry point for the algorithm you'll write for your group project.
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution, 
        time spent to find best solution, total number of solutions found during search, the 
        best solution found.  You may use the other three field however you like.
        algorithm</returns> 
    '''

    def fancy(self, time_allowance=60.0):
        pass

