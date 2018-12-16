import pandas as pd
import numpy as np

from collections import deque
from random import randint
from random import shuffle

class AdjacencyListGraph:
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
    
graph = AdjacencyListGraph('CatBrainEdgeList.dat')  
print(graph.diameter())          

            
            
