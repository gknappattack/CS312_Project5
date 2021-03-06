#!/usr/bin/python3
from queue import PriorityQueue
import copy

from numpy.random import rand

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

SPACE_ALLOWANCE = 3
CLUSTER_SIZE = 10
MIN_CLUSTER_SIZE = 5
MAX_CLUSTER_SIZE = 20


class HeapQueue:
    def __init__(self):
        # Initialize empty array of nodes.
        self.nodes = []

        # Using a dictionary for the array indices to match Node object to location in the heap.
        self.array_indices = {}

        # Max queue size variable used to track the largest queue size each time a node is added.
        self.max_queue_size = 0

    # The delete_min function is essential the same from before for Project 3 with the main change being to the way
    # priority is calculated. Before, the queue was used to sort the nodes by the lowest distance. In this case, the
    # priority key is decided from the result of (node.lowerbound / len(node.path)). Initially I had just used the
    # lower bound as the sorting mechanism for the heap queue, and saw that in testing, I was often getting a good
    # answer rather quickly, but was not pruning many states since the bssf was not updating regularly. To fix this,
    # I added the part about dividing by the length of the current path. The thought behind this was to help push states
    # that may have a slightly higher lower bound above shorter paths with a smaller lowerbound but less cities
    # visited so far. The result was more states being pruned and the bssf being updated more regularly as a result,
    # so I think it accomplished my goal in terms of improvement from just using the lowerbound for the key.
    #
    # The time complexity for this function is O(log n), since as the heap grows bigger, it grows exponentially in size
    # but gets deeper at log(n) rate. Since there are log(n) levels (ex, if there are 64 nodes in the heap, there are
    # log base 2 of 64 = 6 levels of the heap queue). At worst only log(n) traversals can be made, so the time
    # complexity is O(log n).
    def delete_min(self):

        # Parent index is 0
        parent_index = 0  # Initialize variable - Time O(1)

        # Save node at top of heap
        node_to_remove = self.nodes[parent_index]  # Get value from array - Time O(1)

        # Get node at end of heap
        heap_end_node = self.nodes[self.size() - 1]  # Get value from array - Time O(1)

        # Remove node from end of heap
        self.nodes.remove(heap_end_node)  # Remove value from end of array - Time O(1)

        # If node at top was final node,
        if self.size() == 0:  # Check if condition
            del self.array_indices[node_to_remove]  # Delete key, value pair from dictionary - Time O(1)
            return node_to_remove

        # Set top node to final node in heap
        self.nodes[parent_index] = heap_end_node  # Update value in array - Time O(1)

        # Set removed node index to -1
        del self.array_indices[node_to_remove]   # Delete key, value pair from dictionary - Time O(1)

        # Set end nodes index to 0/parent node since it is at top now
        self.array_indices[heap_end_node] = parent_index  # Update value in dictionary for key - Time O(1)

        # Continue to loop until node shifted to top of heap is in correct location
        # Time O(log n), since half the tree is reduced each time the node moves and has log n depth to traverse.
        while True:

            # Get Index of left and right children
            left_child_index = (parent_index * 2) + 1  # Calculate index - O(1)
            right_child_index = (parent_index * 2) + 2  # Calculate index - O(1)

            # Get parent node and lowerbound
            parent_node = self.nodes[parent_index]  # Get value from array - Time O(1)
            parent_lower_bound = parent_node.lowerbound  # Save value from Node object - Time O(1)

            # Get custom priority key using lower bound divided by length of the current path.
            parent_priority = parent_lower_bound / len(parent_node.path)  # Calculate value and save - Time O(1)

            # If left child exists in array
            if left_child_index < self.size():
                left_child_node = self.nodes[left_child_index]  # Get value from array - Time O(1)
                left_child_lower_bound = left_child_node.lowerbound  # Save value from Node object - Time O(1)
                left_child_priority = left_child_lower_bound / len(left_child_node.path)   # Calculate value - Time O(1)
            # If there is no left child (and by extension no right child)
            else:
                left_child_priority = np.inf  # Set value - Time O(1)
                left_child_node = None  # Set value - Time O(1)

            # If right child exists in array
            if right_child_index < self.size():
                right_child_node = self.nodes[right_child_index]  # Get value from array - Time O(1)
                right_child_lower_bound = right_child_node.lowerbound  # Save value from Node object - Time O(1)
                right_child_priority = right_child_lower_bound / len(right_child_node.path) # Calculate value-Time O(1)
            else:
                right_child_priority = np.inf  # Set value - Time O(1)
                right_child_node = None  # Set value - Time O(1)

            # Initialize value to update if switch happens
            parent_swapped = False  # Variable initialization - Time O(1)

            # If left child lowerbound is less than right, use it, tie break left
            if left_child_priority <= right_child_priority:  # Check if condition - Time O(1)
                # If left child is less than parent, update, else do nothing
                if left_child_priority < parent_priority:  # Check if condition - Time O(1)
                    self.nodes[parent_index] = left_child_node  # Update value in array - Time O(1)
                    self.nodes[left_child_index] = parent_node  # Update value in array - Time O(1)

                    self.array_indices[parent_node] = left_child_index  # Update value in array - Time O(1)
                    self.array_indices[left_child_node] = parent_index  # Update value in array - Time O(1)

                    parent_index = left_child_index  # Update variable value - Time O(1)
                    parent_swapped = True  # Update variable value - Time O(1)
            else:  # If right child lowerbound is less than parent, check for swap
                if right_child_priority < parent_priority:  # Check if condition - Time O(1)
                    self.nodes[parent_index] = right_child_node  # Update value in array - Time O(1)
                    self.nodes[right_child_index] = parent_node  # Update value in array - Time O(1)

                    self.array_indices[parent_node] = right_child_index  # Update value in array - Time O(1)
                    self.array_indices[right_child_node] = parent_index  # Update value in array - Time O(1)

                    parent_index = right_child_index  # Update variable value - Time O(1)
                    parent_swapped = True  # Update variable value - Time O(1)

            # If heap did not change on this iteration, break out of while loop and return.
            if not parent_swapped:  # Check if condition - Time O(1)
                break

        # Remove node removed from top of heap queue
        return node_to_remove  # Return value - Time O(1)

    # The decrease_key also works the same as from Project 3 with the exception of the priority key. decrease_key, like
    # delete_min, uses the (node.lowerbound / len(node.path)) formula to find a balance between pushing up nodes with
    # a smaller bound and prioritizing nodes with a longer path in order to update the bssf sooner. Decrease_key is
    # called each time a node is inserted to the heap, moving it up from the last spot in the heap until the priority
    # is greater than the parent node.
    #
    # Same with delete_min, the time complexity of this function is O(log n), since as the heap grows, there are
    # log n levels of depth that a node can travel up or down, making O(log n) an upper bound on the times the while
    # loop of percolating the node.
    def decrease_key(self, node):
        # Check if node has already been popped off of the queue then return early.
        if node not in self.array_indices:  # Check if condition - Time O(1)
            return

        # Get index of child just added to heap queue
        child_index = self.array_indices[node]  # Save value from array - Time O(1)

        # Percolate children up if parent needs to move
        # Time is O(log n) since there are log n levels of depth to the heap, so a worst case/average case
        # may iterate log n times for a larger heap.
        while True:

            # New node reached top of heap, stop percolating
            if child_index == 0:  # Check if condition - Time O(1)
                break

            # Get node from array using index
            child_node = self.nodes[child_index]  # Save value from array - Time O(1)

            # Get parent index using child index - 1 // 2
            parent_index = (child_index - 1) // 2  # Calculate value - Time O(1)
            parent_node = self.nodes[parent_index]  # Save value from array - Time O(1)

            # Create custom priority key using lower bound divided by length of the current path.
            parent_priority = parent_node.lowerbound / len(parent_node.path)  # Calculate value - Time O(1)

            child_priority = child_node.lowerbound / len(child_node.path)  # Calculate value - Time O(1)

            # If the child has a lower priority key than the parent, shift it upwards in the heap.
            if child_priority < parent_priority:  # Check if condition - Time O(1)

                # Switch nodes in the tree for parent and child.
                self.nodes[child_index] = parent_node  # Update value in array - Time O(1)
                self.nodes[parent_index] = child_node  # Update value in array - Time O(1)

                # Update index of each node in array indices dictionary.
                self.array_indices[child_node] = parent_index  # Update value in dictionary - Time O(1)
                self.array_indices[parent_node] = child_index  # Update value in dictionary - Time O(1)

                # Update variable for child index for next iteration.
                child_index = parent_index  # Variable assignment - Time O(1)

            else:  # If the child priority is greater than parent, no change is made and the loop is broken.
                break

    # Function to insert the node to the end of the heap queue, then move it to the right spot by calling decrease_key.
    # Nothing changed from before for this function, except for the addition of a check to update the max_queue_size
    # variable in order to keep track of the largest size of the heap as states are added, removed, and pruned.
    #
    # The actual inserting of the node is O(1) since it is just adding to the end of the array, but the call to
    # decrease_key makes the time complexity of this function O(log n), since it needs to be shifted to the right
    # location in the tree.
    def insert(self, node):
        # Add node to array storage
        self.nodes.append(node)  # Append value to array - Time O(1)

        # Check if queue has new max size after appending new node, update max queue if true.
        if self.size() > self.max_queue_size:  # Check if condition - Time O(1)
            self.max_queue_size = self.size()  # Variable assignment - Time O(1)

        # Save array index to indices dictionary using node as key
        self.array_indices[node] = len(self.nodes) - 1  # Save value to dictionary - Time O(1)

        # Check for shifting node up
        self.decrease_key(node)  # Call to decrease_key - Time O(log n)

    # Return number of nodes in heap queue, used occasionally in the heap functions to get location of end node,
    # and other tasks.
    def size(self):
        return len(self.nodes)  # Return value from len - Time O(1)


# Node is the class used to represent search states in this program. A search state Node has 3 main arrays; the
# cost matrix array that contains the current values for each possible path as they are updated, the path array which
# contains the actual city objects of each city that has been visited that is used by TSPSolution to actually generate
# a solution object, and the cities_visited array which stores the index of the cities that have been visited so far.
# The cities_visited array is not needed for creating the end solution, but since the cost matrix operates in terms of
# source/destination path pairs, it was a lot easier to the index of each city ready to go instead of getting them from
# the cities array every time, even at the cost of additional space to store the indices.
#
# Inside of the node class, there are functions to calculate the cost matrix for a city visited, as well as to check
# and see if a path is a valid solution. The node.lowerbound stores the bound/cost of the current path, and is
# referenced often through the class and branch and bound solution to check for pruning and binding of states.
#
# Time and space complexity of each individual function is given in the comments above each one. In general however,
# functions that solve/reduce the cost matrix are of time complexity O(n^2) since they are iterating over the complete
# n x n array. The overall space complexity of each node is dictated by the same n x n or O(n^2) size array that
# is used for the cost matrix.
class Node:
    # Init function for Node class. For the initial state, most of the values are set to None or zero. However, for
    # each child node, the node is initialized using the cost matrix, path, and cities visited array of the parent
    # node. The cost matrix is copied using deepcopy, which results in the overall time complexity of O(n^2), but
    # simplifies the copying process instead of using nested loops. The other arrays are copied using loops instead
    # of deep copy since deep copy was slowing the code down significantly (30 to 60 seconds in the worst cases).
    #
    # Time complexity - O(n^2) for deep copying the n x n cost matrix.
    # Space complexity - O(n^2) in creating the n x n cost matrix using deep copy. Space is also used to create the
    # path and cities visited arrays, but they are only n length 1d arrays, so the overall space complexity doesn't
    # change.
    def __init__(self, lower_bound, parent_matrix, curr_city, curr_city_index, parent_path, parent_cities):
        # Deep copy cost matrix passed in from parent node.
        self.costmatrix = copy.deepcopy(parent_matrix)  # Deep copy - Time O(n^2)

        # Assign new lower bound to lower bound from parent.
        self.lowerbound = lower_bound

        # Initialize empty path array to save parent cities to
        self.path = []  # Initialize empty array - Time O(1)

        # Copy cities in path from parent - Time O(n)
        for city in parent_path:
            # Append value to array - Time O(1)
            self.path.append(city)

        # Append current city this state represents to the path array - Time O(1)
        self.path.append(curr_city)

        # Initialize empty cities visited array to save cities visited from parent node to.
        self.cities_visited = []  # Initialize empty array - Time O(1)

        # Copy index of cities visited in path from parent - Time O(n)
        for city in parent_cities:
            # Append value to array - Time O(1)
            self.cities_visited.append(city)

        # Append current index of the city this state represents to the path array - Time O(1)
        self.cities_visited.append(curr_city_index)

    # A function that is only used once by the initial state to generate the numpy array used as the cost matrix.
    # This function is not called when creating children states, a deep copy is made by the init function to pass on
    # a new array object in the same current state. This function builds the array from the list of cities passed on
    # from the GUI. The cities are added in the order they are found in the cities list. After creating the array,
    # the cost of each path to and from each city is added to the array at the indices of each source/destination
    # path combination.
    #
    # The array generation itself is an O(n^2) time complexity since it initializes the array of that size with zeroes
    # for each value. Aside from just the array generation, the function then loops through each row and column of the
    # array to check and update the value based on the scenario, which is also a time O(n^2) section of code, since the
    # array has rows and columns equal to the number of cities to visit.
    #
    # Similar to time complexity, the array generated is an n x n array where both length and width are the number
    # of cities to visit, making the space complexity of this function O(n^2)
    def generateCostMatrix(self, cities):
        # Initialize n x n cost matrix using np.zeroes.
        # Time complexity - Time O(n^2) to initialize each values.
        # Space complexity - O(n^2) for n x n array.
        self.costmatrix = np.zeros((len(cities), len(cities)))

        # Loop through each row and column and get the correct value for each path to and from each city.
        # Time complexity O(n^2) - Nested for loop for each row and column of n x n array.
        for sourceindex, city in enumerate(cities):
            for destindex, secondcity in enumerate(cities):

                # If current path checking is to and from same city, set to infinity since there is no path.
                if city == secondcity:  # Check if condition - Time O(1)
                    self.costmatrix[sourceindex][destindex] = np.inf  # Update values in array - Time O(1)
                else:
                    # If path exists, get cost using costTo function and save to array.
                    self.costmatrix[sourceindex][destindex] = city.costTo(secondcity)  # Update value - Time O(1)

    # reduceMatrix is nearly identical in function to the get_initial_lower_bound function below. Both of them
    # center around the behavior of getting the minimum values for the rows and columns in order to reduce the cost
    # matrix for a child state. However, the child states also need to account for the current path being taken as
    # well as account for the cost of the parent path up to this point, making it easier to have the two functions be
    # separate.
    #
    # Although there are a couple additional parts to this function, there is no difference in overall time complexity
    # compared to get_initial_lower_bound, since the nested for loops still dictate the time complexity of this
    # function. Both times, every value is iterated over and updated if needed, making time complexity here O(n^2)
    def reduceMatrix(self, cities, dest_city_index):
        # Get previous city index from cities_visited array
        source_city_index = self.cities_visited[len(self.cities_visited) - 2]  # Get value from array - Time O(1)

        # Get cost of selected path from cost matrix
        path_cost = self.costmatrix[source_city_index][dest_city_index]  # Get value from array - Time O(1)

        # Set path and inverse to infinity in cost matrix
        self.costmatrix[source_city_index][dest_city_index] = np.inf  # Assign value in array - Time O(1)
        self.costmatrix[dest_city_index][source_city_index] = np.inf  # Assign value in array - Time O(1)

        # Set row of source city and column of dest city to inf
        # We update one row and a column together in one for loop, so time is O(n) instead of O(n^2)
        for city_index, city in enumerate(cities):
            self.costmatrix[source_city_index][city_index] = np.inf  # Update value in array - Time O(1)
            self.costmatrix[city_index][dest_city_index] = np.inf  # Update value in array - Time O(1)

        # Get the minimum value of each row using numpy.
        rowminvalues = np.amin(self.costmatrix,
                               axis=1)  # numpy.amin - Time O(n^2) (has to check every value for min)

        # Loop over each row and reduce each value in each column of the row by the min value.
        # Time O(n^2) since each row and column contains every city to visit.
        for rowindex, city in enumerate(self.costmatrix):
            # Get min value from min value by row array
            reducevalue = rowminvalues[rowindex]  # Get value from array - Time O(1)

            # If min value is infinity then set to 0 in min value array to get correct updated lower bound
            if reducevalue == np.inf:  # Check if condition - Time O(1)
                rowminvalues[rowindex] = 0  # Assign value in array - Time O(1)

            for colindex, secondcity in enumerate(self.costmatrix):
                # If value to reduce is 0 or infinity, don't update values in row
                if self.costmatrix[rowindex][colindex] != 0 and reducevalue != np.inf:  # Check if condition - Time O(1)
                    # Update value in current column
                    self.costmatrix[rowindex][colindex] -= reducevalue  # Update variable - Time O(1)

        # Get minimum values for each column using numpy.amin
        colminvalues = np.amin(self.costmatrix, axis=0)  # numpy.amin() - Time O(n^2)

        # Loop over each column and reduce each value in each row of the column by the min value.
        # Time O(n^2) since each row and column contains every city to visit.
        for colindex, city in enumerate(self.costmatrix):
            # Get min value from min value by column array
            reducevalue = colminvalues[colindex]  # Get value from array - Time O(1)

            # If min value is infinity then set to 0 in min value array to get correct updated lower bound
            if reducevalue == np.inf:  # Check if condition - Time O(1)
                colminvalues[colindex] = 0  # Assign value in array - Time O(1)

            for rowindex, secondcity in enumerate(self.costmatrix):
                # If value to reduce is 0 or infinity, don't update values in column
                if self.costmatrix[rowindex][colindex] != 0 and reducevalue != np.inf:  # Check if condition - Time O(1)
                    # Update value in current row
                    self.costmatrix[rowindex][colindex] -= reducevalue  # Update variable - Time O(1)

        # Get the sum of the minimum values used to reduce the matrix
        reduction_cost = np.sum(rowminvalues) + np.sum(colminvalues)  # np.sum for both arrays, Time O(n^2)

        # Get new bound by adding previous lower bound, path cost and reduction cost
        self.lowerbound += reduction_cost + path_cost  # Update variable - Time O(1)

    # The initial function used by the first state created in the branch and bound method to get the starting
    # lower bound value for the rest of the children state to use in their calculations. Functions is extremely
    # similar to reduceMatrix function above, and likely could be combined, but because the cost of the lowerbound
    # of the children nodes rely on the initial lowerbound, the calculations are slightly different, so I ended up
    # having the two functions split since they were developed that way.
    #
    # The cose uses numpy.amin to extract the minimun values from each row or column, then iterates over the
    # entire array to update the values as needed (if they are not infinity or 0). The rows are handled first, since
    # updating the rows may change if there is a need to reduce the column values. For either rows or columns, every
    # value has to be checked and updated, so each nested for loop has a time complexity of O(n^2), since the cost
    # matrix itself has a space complexity of O(n^2), where n is the number of cities to try and visit. Even though
    # there are two sets of nested for loops, the overall time complexity of this function is O(n^2), since every
    # value of the array has to be checked.
    def get_initial_lower_bound(self):
        # Get the minimum value of each row using numpy.
        rowminvalues = np.amin(self.costmatrix, axis=1)  # numpy.amin - Time O(n^2) (has to check every value for min)

        # Loop over each row and reduce each value in each column of the row by the min value.
        # Time O(n^2) since each row and column contains every city to visit.
        for rowindex, city in enumerate(self.costmatrix):
            for colindex, secondcity in enumerate(self.costmatrix):
                # Extract minimun value for this row from rowminvalues array - Time O(1)
                reducevalue = rowminvalues[rowindex]

                # If value to reduce is 0 or infinity, don't update values in row
                if self.costmatrix[rowindex][colindex] != 0 and reducevalue != np.inf:  # Check if condition - Time O(1)
                    # Update value in current column
                    self.costmatrix[rowindex][colindex] -= reducevalue  # Variable assignment - Time O(1)

        # Get minimum values for each column using numpy.amin
        colminvalues = np.amin(self.costmatrix, axis=0)  # numpy.amin() - Time O(n^2)

        # Loop over each column and reduce each value in each row of the column by the min value.
        # Time O(n^2) since each row and column contains every city to visit.
        for colindex, city in enumerate(self.costmatrix):
            for rowindex, secondcity in enumerate(self.costmatrix):
                # Extract minimun value for this column from colminvalues array - Time O(1)
                reducevalue = colminvalues[colindex]

                # If value to reduce is 0 or infinity, don't update values in column
                if self.costmatrix[rowindex][colindex] != 0 and reducevalue != np.inf:  # Check if condition - Time O(1)
                    # Update value in current column
                    self.costmatrix[rowindex][colindex] -= reducevalue  # Variable assignment - Time O(1)

        # Calculate the initial lower bound from the sum of the minimun values used to reduce the rows and columns
        self.lowerbound += np.sum(rowminvalues) + np.sum(colminvalues)  # Sum rows and columns - Time O(n^2)

    # The main function used by the greedy algorithm to calculate what is the shortest path from the current
    # city to any other city in the array. It is a greedy function, because it simply chooses the best path at
    # the time, even if the final solution after that choice may not exist. It is only concerned with getting the
    # lowest cost path from this state.
    #
    # Every time this function is called, it iterates over the list of cities. It does skip over iterations where
    # the city has already been visited, but that does not change the fact that the for loop still iterates for that
    # city to start. This for loop over the array of cities makes the time complexity of this function O(n), where n
    # is the number of cities to visit.
    def get_greedy_path(self, cities):
        # Get the index of the previous city visited
        source_index = self.cities_visited[len(self.cities_visited) - 1]  # Get value from array - Time O(1)

        # Get values of paths to each city for current source city from cost matrix.
        # Slicing row out of array - Time O(n), where n = number of cities to visit (length of array sliced)
        # Space complexity of this array is O(n), as it is taking one row out of the n by n cost matrix array.
        city_row = self.costmatrix[source_index, :]

        # Initialize values to save minimum cost and index to
        dest_index = -1  # Variable initialization - Time O(1)
        min_cost = np.inf  # Variable initialization - Time O(1)

        # For each city in list, check for shortest path and save cost and array. Time O(n)
        for index, city in enumerate(city_row):
            # Skip checking for any cities not visited yet
            if index in self.cities_visited:  # Check if condition - Time O(1)
                continue

            # Check for min value
            if city_row[index] < min_cost:  # Check if condition - Time O(1)
                # Update minimum cost and index values.
                min_cost = city_row[index]  # Variable assignment - Time O(1)
                dest_index = index  # Variable assignment - Time O(1)

        # Check if minimum cost is infinity - If so, there is no valid solution for this path.
        if min_cost == np.inf:  # Check if condition - Time O(1)
            self.lowerbound = min_cost  # Variable assignment - Time O(1)

        # If minimum cost is not infinity, the current path is still okay, update cost and path arrays
        else:
            # Append index of new city visited to array - Time O(1)
            self.cities_visited.append(dest_index)

            # Append city object from cities list to array - Time O(1)
            self.path.append(cities[dest_index])

            # Update current total cost of path saved in lowerbound - Time O(1)
            self.lowerbound += min_cost

            # Check if current path is complete, if so, check for valid path back to starting city.
            if self.test(cities):  # Check if condition - Time O(1)

                # Get cost of path from final city to starting city
                return_path = self.costmatrix[dest_index][self.cities_visited[0]]  # Get value from array - Time O(1)

                # Path to starting node equals infinity, there is no valid return path, don't update solution
                self.lowerbound += return_path  # Variable assignment - Time O(1)

    # A simple function that returns if a path is complete or not. A path is complete if every city is
    # in the array, checked for using the length of the cities array and the path array containing
    # TSPClasses.City objects.
    #
    # It is only a simple len comparison, so the function is O(1) time, no matter what the result is.
    def test(self, cities):
        # Incomplete path, return infinity
        if len(self.path) != len(cities):  # Check if condition - Time O(1)
            return False  # Return false - O(1)

        # If path is complete, return lower bound
        return True  # Return true - O(1)

    # Debugging function used to print cost matrix with indices and tab spacing for easy viewing. This is not used
    # any where in the actual solution, so I did not comment it more specifically, but I did not remove it so it
    # can be used for Project 6 when implementing the final method.
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

        # A greedy algorithm implementation of the TSP solver. While intended for Project 6, I decided to make it earlier
        # to use it to get an initial bssf state for the branch and bound function, since it would get me a decent estimate
        # resulting in more states pruned earlier and less states to create as a result.
        #
        # The function works as described in the specs; it starts from each city and checks the values for all possible
        # paths from the current city to any city not visited and selects the smallest one, adding it to the current path.
        # If it ever encounters a minimum value of infinity, the path is deemed impossible and no solution is added.
        # The function tries this greedy approach for every starting city, updating the best solution from each of the
        # starting points, returning what ever is the best solution.
        #
        # The greedy approach has a time complexity of O(n^3) since there are essentially three nested loops going on.
        # The for loop that determines the starting city iterates over the length of the cities array, giving us O(n) time
        # for the outer-most loop. Inside, there is a while loop that calls the get_greedy_path function until the path
        # is complete (len(path) == len(cities)). While this while loop will break early if the algorithm encounters an
        # invalid solution, on average it will run through the length of the cities array to get a solution, giving us
        # another O(n) time loop. Lastly, the call to get_greedy_path is the third nested loop, since the function itself
        # loops over the length of the cities array to check for the minimum value of the paths out of the current city,
        # resulting in the third O(n) time complexity loop. For a perfect array that has all valid solutions for each
        # starting city, that would give us O(n^3) time complexity as an upper bound for time complexity.
        #
        # Space complexity for the greedy solution is simply O(n^2), since we are not creating many states like in the
        # branch and bound method, just one that has a n x n array for the cost matrix that is referenced the entire time.
    def greedy(self, time_allowance=60.0):
        # Initialize dictionary for results
        results = {}  # Initialize variable - Time O(1)

        # Get list of cities
        cities = self._scenario.getCities()  # O(n) time to get array of cities

        # Save number of cities to variable for easy use.
        ncities = len(cities)  # Get len - Time O(1)

        # Initialize solution count to count number of valid solutions found
        solution_count = 0  # Initialize variable - Time O(1)

        # Initialize solution to infinity to avoid None error
        greedy_solution = None  # Initialize variable - Time O(1)

        # Start timer to check if time allowed is expired.
        start_time = time.time()  # Get time - Time O(1)

        # Try each city as a starting spot for the greedy path algorithm. Try each city.
        # Each city is looped over - Time O(n)
        for starting_city_index in range(ncities):

            # Check if time expired, if so, break and return best solution found so far.
            if time.time() - start_time > time_allowance:  # Check if condition - Time O(1)
                break

            # Initialize starting state for path starting from indicated city.
            # Node.__init__ - Time O(1) since there are no parent arrays to copy
            initial_state = Node(0, [], cities[starting_city_index], starting_city_index, [], [])

            # Generate cost matrix for this state from parent array.
            initial_state.generateCostMatrix(cities)  # Time - O(n^2)

            # Loop until greedy solution finds answer, will loop over all cities if there is a valid solution.
            # Time - O(n) since the while loop will break once every city has been visited.
            while not initial_state.test(cities):

                # Call function get_greedy_path to get min path from current city and add to path.
                initial_state.get_greedy_path(cities)  # Time O(n)

                # There is no valid solution for this matrix
                if initial_state.lowerbound == np.inf:  # Check if statement - Time O(1)
                    break

            # If there is no valid solution yet, add current solution if valid.
            if greedy_solution is None:  # Check if statement - Time O(1)
                # Check that initial solution is valid
                if initial_state.lowerbound != np.inf:  # Check if statement - Time O(1)
                    # Found first valid solution, create TSPSolution and save for greedy_solution.
                    greedy_solution = TSPSolution(initial_state.path)  # Create solution - Time O(n)

                    # Increment number of solutions found.
                    solution_count += 1  # Increment variable - Time O(1)
            else:
                # Greedy solution already exists, check if the new solution is better than previous one.
                if initial_state.lowerbound < greedy_solution.cost:  # Check if statement - Time O(1)
                    # Found a better solution, create TSPSolution and save for greedy_solution.
                    greedy_solution = TSPSolution(initial_state.path)  # Create solution - Time O(n)

                    # Increment number of solutions found.
                    solution_count += 1  # Increment variable - Time O(1)

        # Stop timing for end timer
        end_time = time.time()  # Get time - Time O(1)

        # Save values to results array - Time O(1)
        results['cost'] = greedy_solution.cost if solution_count > 0 else math.inf
        results['time'] = end_time - start_time
        results['count'] = solution_count
        results['soln'] = greedy_solution
        results['max'] = None
        results['total'] = None
        results['pruned'] = None

        # Return result of greedy approach - Time O(1)
        return results

    # The function used by the branch and bound method to expand a node into child states for each possible path
    # out of the current city. The current state is passed in and its values are used to create the new states since
    # the node.__init__ function copies the arrays of the parent node. Expand iterates over the list of cities,
    # checks that the current city is not already on the path of the node. If it is, the city is skipped. If the
    # city has not been visited yet, then a new state is created using the city, which is added to the end of the path.
    # The new state is then passed into the result array T, which is ultimately returned to the main branch and bound
    # method.
    #
    # The time complexity of this function is O(n^2) because of the node creation process. The expand function itself
    # is simply and only relies on a for loop to iterate over the list of cities and create nodes for cities that
    # have not been visited, making it a O(n) function. However, the init function for creating a new node is O(n^2)
    # since the n x n cost matrix has to be copied from the parent state to the new child. This is handled with
    # copy.deepcopy(), which ensures there is not a just a copying of reference to the array, but adds time to the
    # process. The call to reduceMatrix is also O(n^2), ensuring the time complexity of the function to be O(n^2), not
    # O(n)
    #
    # The space complexity of this function is more complicated. Each node created is of size O(n^2) since it contains
    # a unique n x n array for the cost matrix of this unique state. As the states get deeper, less nodes are created
    # per call to expand, but as many as n-1 nodes can be created when expand is called. These nodes are stored in
    # a result array, T, which has n-1 size at its biggest. This makes the space complexity of this function to be
    # O(n^3), since almost n nodes of size n^2 can be created in one call.
    def expand(self, curr_state, cities):
        # Create an empty array to hold all the state nodes created.
        T = []  # Initialize empty array - Time O(1)

        # Iterate over all cities in the list - Time O(n)
        for index, city in enumerate(cities):

            # Only create nodes for cities that have not been visited yet.
            if index not in curr_state.cities_visited:  # Check if statement - Time O(1)
                # Create new state
                new_state = Node(curr_state.lowerbound, curr_state.costmatrix, city, index, curr_state.path,
                                 curr_state.cities_visited)  # Time O(n^2) to copy arrays.

                # Reduce matrix of new state
                new_state.reduceMatrix(cities, index)  # Time O(n^2) - time to reduce cost matrix.

                # Append to result array
                T.append(new_state)  # Append to end of array - Time O(1)

        # Return array of state nodes.
        return T  # Return value - Time O(1)

    # This is the function for the core branch and bound algorithm. The initial state is created from the array of
    # cities from the scenario given by the GUI. The branch and bound solution is always started from city at index 0.
    # After the initial state's cost matrix and path is handled and the starting lower bound is found, the heap queue
    # is created and an initial bssf using the greedy algorithm is found. From there, the looping begins;
    # for each iteration, the state with the lowest priority key is popped from the heap and expanded to child states.
    # Each of the child states is check for a valid solution. If a solution is found, then the cost is compared to the
    # bssf. If the cost is an improvement, the solution is added. If the current state is not a complete solution, it
    # is checked against the bound for pruning. If the current state is under the bound, it is pushed onto the heap,
    # else nothing happens and the state is pruned. Each iteration of the while loop checks the elapsed time, breaking
    # the loop at its head if the expected time allowance is passed.
    #
    # The time complexity of the function is still the exponential time of O(2^n). The branching and bounding solution
    # ensures that we get closer to a solution by not spending as much time working on states that do not go anywhere.
    # That does not change the fact that we create all of the states using the expand function in order to check them
    # for pruning. Each state results in an exponentially growing number of children states until a complete solution
    # is found. The pruning reduces the number of states created, especially when a state exceeds the bound early into
    # the path. The greedy algorithm returns a fairly lower bssf, which results in slightly higher early pruning
    # than compared to when I ran branch and bound with the random solution. Still, as the total number of states
    # created in my testing indicates, the number of states created is still an exponential number, making the while
    # loop continue for an exponential time. Of course in our code, we use a time limit to stop the program after
    # a max of one minute, but if the code were to continue, the run time would take a while since it has to loop
    # until there are no more states left on the queue to loop through. Since every state create is a loop on the for
    # loop, the time complexity is O(2^n)
    #
    # The space complexity for branch and bound is a bit more complicated. While the time is exponential, the space
    # complexity of the function is not because of the pruning used. As mentioned in the comment above the heap queue,
    # in testing, it was rare to see a heap queue max size result greater than O(n^2), with n being the number of
    # cities to visit. Since the nodes contain the n x n cost matrix and each have a space of O(n^2), then we can
    # say the heap queue has a space complexity of O(n^4), assuming the upper bound of the queue size is O(n^2).
    # The nodes are still created by the expand function, which is as mentioned in the comments there,
    # space complexity of O(n^3) since the nodes have a n x n array, and up to n - 1 nodes are created. If we consider
    # the heap queue to have a space complexity upper bound of O(n^2) based on the max queue size results, and the
    # expand function to return an array T of size O(n) containing nodes of size O(n^2), then our space complexity
    # of the branch and bound function would be O(n^4), since the O(n^4) size of the heap queue with nodes at max size
    # would dominate the O(n^3) size of the T array containing nodes created by expand.
    def branchAndBound(self, time_allowance=60.0):
        # Create empty results array
        results = {}  # Create empty array - Time O(1)

        # Get list of cities from scenario
        cities = self._scenario.getCities()  # Get cities - Time O(1)

        # Initialize the initial state starting from city A
        initial_state = Node(0, [], cities[0], 0, [], [])  # Time O(1) since there are no arrays to copy

        # Create initial cost matrix for initial state
        initial_state.generateCostMatrix(cities)  # Create initial matrix - Time O(n^2)

        # Reduce and calculate initial lower bound
        initial_state.get_initial_lower_bound()  # Reduce initial matrix - Time O(n^2)

        # Create empty Heap Queue and insert initial state
        S = HeapQueue()  # Create Heap Queue - Time O(1)
        S.insert(
            initial_state)  # Insert initial state to heap queue - Time O(1) since there is nothing in the queue

        # Initialize variables used in results array)
        solution_count = 0  # Initialize variable - Time O(1)
        states_pruned = 0  # Initialize variable - Time O(1)
        total_states = 1  # Initialize variable - Time O(1)

        # Get starting bssf using greedy algorithm.
        greedy_result = self.greedy()  # Greedy algorithm solution - Time O(n^3)
        bssf = greedy_result['soln']  # Save value from dictionary - Time O(1)

        # Initialize timer to check for time out
        start_time = time.time()  # Get time - Time O(1)

        # While Heap Queue is not empty and time allowed has not passed - Time Complexity (See function analysis above)
        while S.size() > 0 and time.time() - start_time < time_allowance:

            # Get node at top of heap
            P = S.delete_min()  # Delete min from Heap Queue - Time O(log n)

            # Expand P into sub states, add to T
            T = []  # Create empty list - Time O(1)
            T = self.expand(P, cities)  # Expand function - Time O(n^2)

            # Update number of total states created by the expand function
            total_states += len(T)  # Variable Assignment - Time O(1)

            # For each state in T, check if is a solution or if it should be added to heap queue
            for Pi in T:  # Expand creates at most n - 1 nodes, so for loop is Time O(n)

                # Reached a solution, compare cost to bssf
                if Pi.test(cities):  # Node.test - Time O(1)

                    # If the current solution has a lower cost than bssf, replace bssf
                    if Pi.lowerbound < bssf.cost:  # Check if condition - Time O(1)
                        bssf = TSPSolution(Pi.path)  # Create solution - Time O(n)

                        # Increment solution count since bssf was updated
                        solution_count += 1  # Variable assignment - Time O(1)

                # Not at a solution yet, check if state should be added to queue or pruned
                else:
                    # Lower bound of partial solution is >= BSSF so prune the current state.
                    if Pi.lowerbound >= bssf.cost:  # Check if condition - Time O(1)

                        # Update count of states pruned
                        states_pruned += 1  # Variable assignment - Time O(1)

                    # Path lower bound is < BSSF push to heap queue and wait for next round
                    else:
                        S.insert(Pi)  # Insert node to heap queue - Time O(log n)

        # Stop timing for end timer
        end_time = time.time()  # Get time - Time O(1)

        # Push results of bssf and other variables to results array - Time O(1)
        results['cost'] = bssf.cost if solution_count > 0 else math.inf
        results['time'] = end_time - start_time
        results['count'] = solution_count
        results['soln'] = bssf
        results['max'] = S.max_queue_size
        results['total'] = total_states
        results['pruned'] = states_pruned

        # Return results - Time O(1)
        return results

    def fancy(self, time_allowance=60.0):
        results = {}
		cities = self._scenario.getCities()
		# children = cluster_all_children(cities)
		cluster_prio = PriorityQueue()
		final_route = []
		not_valid_solution = True
		# final_route_2 = []
		# final_route_3 = []
		cluster_cost_matrix = None
		cluster_connection_matrix = None

		time_start = time.time()

		while(not_valid_solution):
			print("start")
			final_route.clear()
			not_valid_solution = False

			children, kworked = Kmeans(cities)
			if not kworked:
				not_valid_solution = True
				continue

			print("children done")
			cluster_cost_matrix, cluster_connection_matrix = find_children_connections(children)
			print("made clusters")
			cluster_route, pruned_states = branch_and_bound_algorithm(len(children), cluster_cost_matrix, 0)
			print("made route")
			route_conn = select_connection_tuple(cluster_route, cluster_connection_matrix)

			for i in range(len(cluster_route)):
				print(i, end=".")
				child = children[cluster_route[i]]
				childMatrix = initialize_matrix(child)
				start_cluster = cluster_route[i-1]
				end_cluster = cluster_route[0] if i == len(cluster_route)-1 else cluster_route[i+1]
				start_city = route_conn[i-1][1]
				end_city = route_conn[i][0]
				if end_city not in child:
					continue
				if start_city not in child:
					continue
				start_city_idx = child.index(start_city)
				end_city_idx = child.index(end_city)
				# print(i, "cluster: ", start_city._name, ", ", end_city._name)

				setup_start_and_end(childMatrix, start_city_idx, end_city_idx)
				child_route, just_pruned = branch_and_bound_algorithm(len(child), childMatrix, start_city_idx)
				if just_pruned == np.inf:
					print([city._name for city in child])
					not_valid_solution = True
					break
				else:
					pruned_states += just_pruned
				# final_route_2.extend(child_route.copy())
				# final_route_3.extend([child[idx]._name for idx in child_route])
				final_route.extend([child[idx] for idx in child_route])

			if not_valid_solution == True:
				continue

		time_end = time.time()
		print([my_child._name for my_child in final_route])
		#wrap-up
		bssf = TSPSolution(final_route)
		results['cost'] = bssf.cost
		results['time'] = time_end - time_start
		results['count'] = 1
		results['soln'] = bssf
		results['max'] = 100
		results['total'] = 100
		results['pruned'] = pruned_states
		return results


def find_child_val(start_city_index, next_city_index, cost_matrix_orig, prev_lowerbound):
    init_cost = cost_matrix_orig[start_city_index][next_city_index]

    cost_matrix = copy.deepcopy(cost_matrix_orig)
    for i in range(len(cost_matrix[start_city_index])):
        cost_matrix[start_city_index][i] = np.inf
    for j in range(len(cost_matrix)):
        cost_matrix[j][next_city_index] = np.inf
    cost_matrix[next_city_index][start_city_index] = np.inf

    lowerbound = prev_lowerbound + reduce_cost_matrix(cost_matrix) + init_cost
    return lowerbound, cost_matrix


def reduce_cost_matrix(matrix):
    total_reduced = 0
    for i in range(len(matrix)):
        min_row = min(matrix[i])
        if min_row != 0 and min_row != np.inf:
            for j in range(len(matrix[i])):
                matrix[i][j] -= min_row
            total_reduced += min_row

    for j in range(len(matrix[0])):
        min_col = min([i[j] for i in matrix])
        if min_col != 0 and min_col != np.inf:
            for i in range(len(matrix)):
                matrix[i][j] -= min_col
            total_reduced += min_col

    return total_reduced


def initialize_matrix(cities):
    matrix = [[0 for i in range(len(cities))] for i in range(len(cities))]
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            matrix[i][j] = cities[i].costTo(cities[j])
    return matrix


class State:
    def __init__(self, curr_route, curr_cost_matrix, curr_lowerbound):
        self.route = copy.deepcopy(curr_route)
        self.cost_matrix = curr_cost_matrix
        self.lowerbound = curr_lowerbound


def cluster_all_children(cities):
    unused_cities = [cities[i] for i in range(len(cities))]
    cluster_list = []
    i = 0
    cluster_dict = dict()
    while len(unused_cities) > 0:
        print(i, ": ")
        cluster_list.append(cluster_child(get_start_city(unused_cities), unused_cities, cluster_dict, i))
        i += 1

    return cluster_list


def cluster_child(start_city, unused_cities, cluster_dict, iter):
    pqueue = PriorityQueue()
    for i in range(len(unused_cities)):
        pqueue.put((start_city.costTo(unused_cities[i]), rand(), unused_cities[i]))
    if pqueue.qsize() > 0:
        _, _, next_city = pqueue.get()
        for i in range(len(unused_cities)):
            pqueue.put((next_city.costTo(unused_cities[i]), rand(), unused_cities[i]))
    if pqueue.qsize() > 0:
        _, _, next_city = pqueue.get()
        for i in range(len(unused_cities)):
            pqueue.put((next_city.costTo(unused_cities[i]), rand(), unused_cities[i]))
    if pqueue.qsize() > 0:
        _, _, next_city = pqueue.get()
        for i in range(len(unused_cities)):
            pqueue.put((next_city.costTo(unused_cities[i]), rand(), unused_cities[i]))

    child_set = set()
    while len(child_set) < CLUSTER_SIZE and len(child_set) < len(unused_cities):
        _, _, to_add = pqueue.get()
        child_set.add(to_add)

    child_set_cpy = child_set.copy()

    for i in range(len(child_set)):
        city = child_set_cpy.pop()
        unused_cities.remove(city)
        cluster_dict[city] = iter

    child_set_list = set_to_list(child_set.copy())
    print([child._name for child in child_set])
    return set_to_list(child_set)


def find_children_connections(children):
    cluster_connection_matrix = [[(None, None) for i in range(len(children))] for j in range(len(children))]
    cluster_cost_matrix = [[np.inf for i in range(len(children))] for j in range(len(children))]

    child_cnt = len(children)

    if child_cnt == 2:
        cost, conn_1 = shortest_connections_for_two(children[0], children[1])
        cost, conn_2 = shortest_connections_for_two(children[1], children[0])
        cluster_cost_matrix[0][1] = cost
        cluster_connection_matrix[0][1] = conn_1
        cluster_cost_matrix[1][0] = cost
        cluster_connection_matrix[1][0] = conn_2
    else:
        for i in range(child_cnt):
            for j in range(child_cnt):
                if i == j:
                    continue
                cost, conn_1 = shortest_connection(children[i], children[j])
                cluster_cost_matrix[i][j] = cost
                cluster_connection_matrix[i][j] = conn_1

    return cluster_cost_matrix, cluster_connection_matrix


def shortest_connection(city_list1, city_list2):
    prioQueue = PriorityQueue()
    avg = 0

    for city in city_list1:
        for city_to in city_list2:
            curr_cost = city.costTo(city_to)
            avg = avg + curr_cost if curr_cost != np.inf else avg + 500
            conn = (city, city_to)
            prioQueue.put((curr_cost, hash(conn), conn))
    avg = avg / (len(city_list1) * len(city_list2))

    return avg, prioQueue


def shortest_connections_for_two(city_list1, city_list2):
    min_1 = np.inf
    min_2 = np.inf
    conn_1 = None
    conn_2 = None
    prioQueue = PriorityQueue()

    for city in city_list1:
        for city_to in city_list2:
            curr_cost = city.costTo(city_to)
            prioQueue.put((curr_cost, hash((city, city_to)), (city, city_to)))

    # min_1, _, conn_1 = prioQueue.get()
    # min_2, _, conn_2 = prioQueue.get()
    # while(conn_has_overlap(conn_2, conn_1)):
    # 	min_2, _, conn_2 = prioQueue.get()

    return 0, prioQueue


def conn_has_overlap(conn_1, conn_2):
    if conn_2 is None:
        return False
    for city in conn_2:
        for other_city in conn_1:
            if city == other_city:
                return True

    return False


def select_connection_tuple(cluster_route, cluster_conn_matrix):
    used_cities = []
    route_conn = []
    for i in range(len(cluster_route)):
        prev_route_len = len(route_conn)
        curr = cluster_route[i]
        next = cluster_route[0] if i == len(cluster_route) - 1 else cluster_route[i + 1]
        poss_connections = cluster_conn_matrix[curr][next]
        while prev_route_len == len(route_conn):
            conn = poss_connections.get()[2]
            if conn[0] not in used_cities and conn[1] not in used_cities:
                route_conn.append(conn)
                used_cities.append(conn[0])
                used_cities.append(conn[1])
                break
        print(i, ":", len(route_conn))

    return route_conn


def get_start_city(set):
    for e in set:
        break
    return e


def set_to_list(set):
    set_cpy = set.copy()
    list = []
    while (len(set_cpy) != 0):
        list.append(set_cpy.pop())
    return list


def min_conn(child_idx, cost_matrix, conn_matrix):
    min_row = np.inf
    min_col = np.inf
    min_col_val = None
    min_row_val = None
    for i in range(len(cost_matrix)):
        curr_row = cost_matrix[child_idx][i]
        curr_col = cost_matrix[i][child_idx]
        if curr_row < min_row:
            min_row = curr_row
            min_row_val = conn_matrix[child_idx][i]
            row_y = i
            if conn_matrix[row_y][child_idx] < min_col:
                min_col = conn_matrix[row_y][child_idx]
                min_col_val = conn_matrix[row_y][child_idx]
        elif curr_col < min_col:
            min_col = curr_col
            min_col_val = conn_matrix[i][child_idx]

    return min_row, min_row_val, min_col, min_col_val


def branch_and_bound_algorithm(cities_cnt, cityMatrix, start_c):
    lowerbound = reduce_cost_matrix(cityMatrix)
	my_route = []
	my_route.append(start_c)
	init_city = start_c
	prioQueue = PriorityQueue()
	init_state = State(my_route, cityMatrix, lowerbound)
	prioQueue.put((lowerbound, my_route, init_state))
	waitQueue = PriorityQueue()
	foundTour = False
	pruned_states = 0

	while not foundTour:
		if prioQueue.qsize() < 1:
			print("error")
			return [], np.inf
		lowerbound, curr_path, curr_state = prioQueue.get()
		start_city = curr_path[-1]
		my_route = curr_state.route
		cityMatrix = curr_state.cost_matrix
		lowerbound = curr_state.lowerbound

		for i in range(cities_cnt):
			if i != init_city:
				if cityMatrix[start_city][i] != np.inf:
					child_val, child_matrix = find_child_val(start_city, i, cityMatrix, lowerbound)
					my_route.append(i)
					child_state = State(my_route, child_matrix, child_val)
					waitQueue.put((child_val//len(my_route), my_route.copy(), child_state))
					my_route.pop()
		for i in range(SPACE_ALLOWANCE):
			if waitQueue.qsize() > 0:
				prioQueue.put(waitQueue.get())
		pruned_states += waitQueue.qsize()
		waitQueue.queue.clear()


		# TODO: len(cities)-1?
		if len(my_route) == cities_cnt:
			foundTour = True


	return my_route, pruned_states


def setup_start_and_end(childMatrix, start, end):
    for i in range(len(childMatrix)):
        childMatrix[i][start] = np.inf
        childMatrix[end][i] = np.inf
    childMatrix[start][end] = np.inf


def Kmeans(cities):
	cities_cnt = len(cities)
	kcnt = cities_cnt//CLUSTER_SIZE + 1
	cluster_list = [Cluster() for k in range(kcnt)]
	isUpdating = True
	kWorked = True
	i = 0
	while isUpdating:
		print(i, end=",")
		i+=1
		if i > 75:
			kWorked = False
			print("kmeans runtime error")
			return [], kWorked

		isUpdating = False
		assign_cities_to_cluster(cities, cluster_list)
		for cluster in cluster_list:
			if len(cluster.cities) < MIN_CLUSTER_SIZE:
				dissolve_cluster(cluster, cluster_list)
				unchanged = False
				continue
			if len(cluster.cities) > MAX_CLUSTER_SIZE:
				split_cluster(cluster, cluster_list)
				unchanged = False
				continue
			unchanged = cluster.update()
			if not unchanged:
				isUpdating = True

	assign_cities_to_cluster(cities, cluster_list)
	return [cluster.cities for cluster in cluster_list], kWorked


def assign_cities_to_cluster(cities, cluster_list):
    for city in cities:
        min_cluster = None
        min_val = np.inf
        for cluster in cluster_list:
            dist = cluster.distance(city)
            if dist < min_val:
                min_cluster = cluster
                min_val = dist
        min_cluster.add_city(city)


class Cluster():
    def __init__(self, xy=None):
        self.data_range = {'x': [-1.5, 1.5], 'y': [-1.0, 1.0]}
        if xy is not None:
            self._x = xy[0]
            self._y = xy[1]
            self.loc = None
        else:
            self._x = None
            self._y = None
            self.loc = self.random_assignment()

        self.old_loc = None
        self.cities = []

    def random_assignment(self):
        xr = self.data_range['x']
        yr = self.data_range['y']
        x = random.uniform(0.0, 1.0)
        y = random.uniform(0.0, 1.0)
        if True:
            xval = xr[0] + (xr[1] - xr[0]) * x
            yval = yr[0] + (yr[1] - yr[0]) * y
            self._x = xval
            self._y = yval
            return QPointF(xval, yval)

    def update(self):
        self.old_loc = (self._x, self._y)
        self.loc = self.computeCentroid()
        cost = math.sqrt((self._x - self.old_loc[0]) ** 2 +
                         (self._y - self.old_loc[1]) ** 2)

        tot = int(math.ceil(cost * self.MAP_SCALE))
        return tot == 0

    MAP_SCALE = 1000.0

    def distance(self, other_city):
        # Euclidean Distance
        cost = math.sqrt((other_city._x - self._x) ** 2 +
                         (other_city._y - self._y) ** 2)

        return int(math.ceil(cost * self.MAP_SCALE))

    def computeCentroid(self):
        if len(self.cities) == 0:
            return self.loc
        yval = 0
        xval = 0
        for city in self.cities:
            yval += city._y
            xval += city._x
        yval /= len(self.cities)
        xval /= len(self.cities)
        self._x = xval
        self._y = yval
        self.cities.clear()
        return QPointF(xval, yval)

    def clearCities(self):
        self.cities.clear()

    def add_city(self, city):
        self.cities.append(city)

    def change_loc(self, xy):
        self._x = xy[0]
        self._y = xy[1]


def dissolve_cluster(cluster, cluster_list):
    cluster_list.remove(cluster)
    assign_cities_to_cluster(cluster.cities, cluster_list)


def split_cluster(cluster, cluster_list):
    std_dev_x = 0
    std_dev_y = 0
    for city in cluster.cities:
        std_dev_x += abs(cluster._x - city._x)
        std_dev_y += abs(cluster._y - city._y)

    std_dev_x /= len(cluster.cities)
    std_dev_y /= len(cluster.cities)

    split_x = cluster._x + std_dev_x
    split_y = cluster._y + std_dev_y
    other_x = cluster._x - std_dev_x
    other_y = cluster._y - std_dev_y

    new_cluster = Cluster((split_x, split_y))
    cluster.change_loc((other_x, other_y))
    cluster_list.append(new_cluster)
    cluster.clearCities()
