import pandas as pd
import numpy as np
import time

from collections import deque
from random import randint
from random import shuffle

class AdjacencyListGraph:
    white = 0
    grey = 1
    black = 2
    
    def __init__(self, filename, delimiter='  ', undirected=True):
        self.degree_count = 0
        self.edge_count = 0
        self.node_count = 0
        self.adjacency_list = None
        self.read_adjacency_list(filename, undirected=undirected, 
                                 delimiter=delimiter)

    def read_adjacency_list(self, filename, undirected, delimiter):
        source = 'source'
        target = 'target'
        df = pd.read_csv(filename, delimiter=delimiter, names=[source, target], 
                          engine='python')
        
        node2idx = {}
        idx = 0
        for i, row in df.iterrows():
            src = row[source]
            tgt = row[target]
            
            if node2idx.get(src) == None:
                node2idx[src] = idx
                idx += 1
                
            if node2idx.get(tgt) == None:
                node2idx[tgt] = idx
                idx += 1
        
        source_max = (df[source].max())
        target_max = (df[target].max())
        self.node_count = source_max if source_max > target_max else target_max
        self.adjacency_list = [[] for _ in range(self.node_count)]
        for i, row in df.iterrows():
            src = node2idx[row[source]]
            tgt = node2idx[row[target]]
            
            self.adjacency_list[src].append(tgt)
            
            if undirected:
                self.adjacency_list[tgt].append(src)
            
            self.edge_count += 1
        
        self.adjacency_list = [sorted(row) for row in self.adjacency_list]
        self.degree_count = self.edge_count * 2
    
    # NETWORK TRANSFORMATIONS
        
    def to_gnm_network(self):
        gnm = [[] for _ in range(self.node_count)]
        
        for i in range(self.edge_count):
            src = randint(0, self.node_count - 1)
            tgt = randint(0, self.node_count - 1)
            gnm[src].append(tgt)
            gnm[tgt].append(src)
        
        return [sorted(row) for row in gnm] 
    
    def to_configuration_model(self):
        stubs = []
        for i, node in enumerate(self.adjacency_list):
            degree = len(node)
            stubs.append([i] * degree)    
        stubs = [i for row in stubs for i in row] # flatten
        shuffle(stubs)
        
        configuration_model = [[] for _ in range(self.node_count)]
        for i in range(0, len(stubs), 2):
            src = stubs[i]
            tgt = stubs[i+1]
            configuration_model[src].append(tgt)
            configuration_model[tgt].append(src)
        
        return [sorted(row) for row in configuration_model]
    
    # DEGREE DISTRIBUTIONS
    
    def degree_distribution(self):
        return [len(row) for row in self.adjacency_list]
    
    def normalised_degree_distribution(self):
        return [len(row)/self.node_count for row in self.adjacency_list]
    
    def variance_degree_distribution(self):
        return np.array(self.degree_distribution()).var()
    
    def mean_degree_distribution(self):
        return np.array(self.degree_distribution()).mean()

    # CLUSTERING COEFFICIENTS
    
    def local_clustering_coefficients(self):
        clustering_coefs = [0.0 for _ in range(self.node_count)]
        
        for node in range(self.node_count):
            triads = 0
            triangles = 0
            
            for nb in range(len(self.adjacency_list[node])):
                neighbour = self.adjacency_list[node][nb]
                
                # skip reflexive edges (self-loops), which cannot form triads
                is_reflexive_edge = (node == neighbour)
                if is_reflexive_edge:
                    continue
                
                # skip duplicate edges. this assumes every vector within 
                # adjacency list is sorted
                is_duplicate_edge = (neighbour == self.adjacency_list[node][nb - 1])
                if nb != 0 and is_duplicate_edge:
                    # since nb =0 is the first neighbour in the adjacency list
                    # for that particular node, it cannot be a duplicate edge
                    continue
    
                # loop over higher neighbours
                degree = len(self.adjacency_list[node])
                for high in range(nb+1, degree):
                    higher_neighbour = self.adjacency_list[node][high]
                    triads += 1
                    
                    if higher_neighbour in self.adjacency_list[neighbour]:
                        triangles += 1
            
            clustering_coefs[node] = triangles / triads if triangles != 0 and triads != 0 else 0
    
        return clustering_coefs

    def mean_local_clustering_coefficient(self):
        return np.array(self.local_clustering_coefficients()).mean()
    
    # ASSORTATIVITY COEFFICIENT
    def assortativity_coefficient(self):
        mean = self.mean_excess_degree()
        variance = self.variance_excess_degree()
        
        excess_KiKj = 0
        for node in range(self.node_count):
            degree_Ki = len(self.adjacency_list[node])
            
            for nb in range(len(self.adjacency_list[node])):
                neighbour = self.adjacency_list[node][nb]
                degree_Kj = len(self.adjacency_list[neighbour])
                
                excess_Ki = degree_Ki - 1
                excess_Kj = degree_Kj - 1
                excess_KiKj = (excess_Ki * excess_Kj)
    
        return ((excess_KiKj / self.degree_count) - mean ** 2) / variance
    
    def mean_excess_degree(self):
        total = 0
        
        for node in self.adjacency_list:
            degree = len(node)
            excess_degree = degree - 1
            total += (degree * excess_degree)
        
        return total / self.degree_count
    
    def variance_excess_degree(self):
        variance = 0
        mean = self.mean_excess_degree()
        
        for node in self.adjacency_list:
            degree = len(node)
            excess_degree = degree - 1
            variance += ((excess_degree - mean) ** 2) * degree
        
        return variance / self.degree_count
    
    # PATH FINDING
    
    def all_shortest_path_lengths(self):
        return [self.shortest_path_lengths(i) for i in range(self.node_count)]
        
    def shortest_path_lengths(self, source):
        distances = [-1] * self.node_count
        distances[source] = 0
        
        q = deque([source])
        
        while q:
            node = q.pop()
            distance = distances[node]
            
            for neighbour in self.adjacency_list[node]:
                if distances[neighbour] == -1:
                    distances[neighbour] = distance + 1
                    q.appendleft(neighbour)
        
        return distances
    
    def mean_shortest_path_length(self):
        all_distances = self.all_shortest_path_lengths()
        
        total = 0
        total_path_count = 0
        
        for distances in all_distances:
            for d in distances:
                if d > 0: # exclude starting node and unvisited nodes
                    total += d
                    total_path_count += 1
        
        return total / total_path_count
    
    def diameter(self):
        all_distances = self.all_shortest_path_lengths()
        
        diameter = -1
        for distances in all_distances:
            for d in distances:
                if d > diameter:
                    diameter = d
        
        return diameter    
    
    # CLOSENESS CENTRALITY
    
    def all_closeness_centrality(self):
        return [self.closeness_centrality(node) for node in range(self.node_count)]
    
    def closeness_centrality(self, node):
        distances = self.shortest_path_lengths(node)
        
        total = 0
        num_distances = 0
        
        for d in distances:
            if d > 0:
                total += d
                num_distances += 1
        
        return num_distances / total
    
    # LOCAL SEARCH
    
    def self_avoiding_degree_seeking_local_search(self, source, target):
        current = source
        colours = [self.white] * self.node_count
        local_search = []
        
        while current != target:
            local_search.append(current)
            self.__set_colours(colours, current)
            current = self.__target_or_random_highest_degree_white_grey_neighbour(colours, 
                                                                                  current, 
                                                                                  target)
        local_search.append(current)
        return local_search
    
    def __set_colours(self, colours, current):
        is_all_visited = True
        for neighbour in self.adjacency_list[current]:
            if colours[neighbour] == self.white:
                is_all_visited = False
                break
        
        colours[current] = self.black if is_all_visited else self.grey
    
    def __target_or_random_highest_degree_white_grey_neighbour(self, colours, current, target):
        # stable sort used to ensure randomnessis maintained where 
        # there are multiple nodes of the same degree
        neighbours = shuffle(self.adjacency_list[current])
        neighbours.sort(key=lambda x, y: len(self.adjacency_list[x]) > len(self.adjacency_list[y]))
        
        white_neighbour = None
        grey_neighbour = None
        target_neighbour = None
        
        for neighbour in neighbours:
            colour = colours[neighbour]
            
            if neighbour == target:
                return target
            
            if colour == white: 
                white_neighbour = neighbour
            
            if colour == grey: 
                grey_neighbour = neighbour
            
        if white_neighbour != None:
            return white_neighbour
        elif grey_neighbour != None:
            return grey_neighbour
        
    def degree_biased_random_walk(self, source, target):
        current = source
        random_walk = []
        
        while current != target:
            random_walk.append(current)
            
            is_target_neighbour = current in self.adjacency_list[target]
            current = target if is_target_neighbour else self.__random_highest_degree_neighbour(current)
        
        random_walk.append(current) # target node
        return random_walk
    
    def degree_biased_random_walk_by_duration(self, source, duration_seconds):
        current = source
        random_walk = []
        
        start = time.time()
        elapsed = 0
        while elapsed < duration_seconds:
            current = self.__random_highest_degree_neighbour(current)
            random_walk.append(current)
            
            current = time.time()
            elapsed = current - start
        
        return random_walk
    
    def __random_highest_degree_neighbour(self, current):
        neighbours = self.__degree_of_neighbours(current)
        random_neighbour = randint(0, len(neighbours) - 1)
        return neighbours[random_neighbour]
    
    def __degree_of_neighbours(self, current):
        return [len(self.adjacency_list[neighbour]) for neighbour in self.adjacency_list[current]]
        
    # ONION AND K-CORE DECOMPOSITION
    
graph = AdjacencyListGraph('CatBrainEdgeList.dat')       

            
            
