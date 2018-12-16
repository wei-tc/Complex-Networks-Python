import pandas as pd
import numpy as np

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
        
        self.degree_count = self.edge_count * 2
        
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
            num_neighbours = len(node)
            stubs.append([i] * num_neighbours)    
        stubs = [i for row in stubs for i in row] # flatten
        shuffle(stubs)
        
        configuration_model = [[] for _ in range(self.node_count)]
        for i in range(0, len(stubs), 2):
            src = stubs[i]
            tgt = stubs[i+1]
            configuration_model[src].append(tgt)
            configuration_model[tgt].append(src)
        
        return [sorted(row) for row in configuration_model]
    
    def degree_distribution(self):
        return [len(row) for row in self.adjacency_list]
    
    def normalised_degree_distribution(self):
        return [len(row)/self.node_count for row in self.adjacency_list]
    
    def variance_degree_distribution(self):
        return np.array(self.degree_distribution()).var()
    
    def mean_degree_distribution(self):
        return np.array(self.degree_distribution()).mean()
    
graph = AdjacencyListGraph('CatBrainEdgeList.dat')
print(graph.mean_degree_distribution())

            
            
            
