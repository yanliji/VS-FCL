import numpy as np
from . import tools

num_node = 25
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                    (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                    (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                    (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
external = ([(4, 12), (8, 4), (12, 8), (12, 9), (8, 15)])
neighbor = inward + outward

num_node1 = 20
self_link1 = [(i, i) for i in range(num_node1)]
inward_ori_index1 = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18), (20, 19), (22, 23), (23, 8)]
inward1 = [(i - 1, j - 1) for (i, j) in inward_ori_index1]
outward1 = [(j, i) for (i, j) in inward1]
external1 = ([(8, 4), (8, 15)])
neighbor1 = inward1 + outward1

num_node2 = 20
self_link2 = [(i, i) for i in range(num_node2)]
inward_ori_index2 = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18), (20, 19), (24, 25), (25, 12)]
inward2 = [(i - 1, j - 1) for (i, j) in inward_ori_index2]
outward2 = [(j, i) for (i, j) in inward2]
external2 = ([(4, 12), (12, 9)])
neighbor2 = inward2 + outward2

num_node3 = 22
self_link3 = [(i, i) for i in range(num_node3)]
inward_ori_index3 = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (22, 23), (23, 8), (24, 25), (25, 12)]
inward3 = [(i - 1, j - 1) for (i, j) in inward_ori_index3]
outward3 = [(j, i) for (i, j) in inward3]
external3 = ([(4, 12), (8, 4), (12, 8), (12, 9), (8, 15)])
neighbor3 = inward3 + outward3

num_node4 = 22
self_link4 = [(i, i) for i in range(num_node4)]
inward_ori_index4 = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1), (17, 1), (18, 17), (19, 18), (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
inward4 = [(i - 1, j - 1) for (i, j) in inward_ori_index4]
outward4 = [(j, i) for (i, j) in inward4]
external4 = ([(4, 12), (8, 4), (12, 8), (12, 9)]) 
neighbor4 = inward4 + outward4

num_node5 = 20
self_link5 = [(i, i) for i in range(num_node5)]
inward_ori_index5 = [(6, 5), (7, 6), (8, 7), (10, 9), (11, 10), (12, 11), (14, 13), (15, 14), (16, 15), (18, 17), (19, 18), (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
inward5 = [(i - 1, j - 1) for (i, j) in inward_ori_index5]
outward5 = [(j, i) for (i, j) in inward5]
external5 = ([(12, 8), (12, 9), (8, 15)])
neighbor5 = inward5 + outward5


class Graph():
    """
    Arguments:
        labeling_mode: must be one of the follow candidates
            uniform: Uniform Labeling
            dastance*: Distance Partitioning*
            dastance: Distance Partitioning
            spatial: Spatial Configuration
            DAD: normalized graph adjacency matrix
            DLD: normalized graph laplacian matrix
    """

    def __init__(self, labeling_mode='spatial'):
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'uniform':
            A = tools.get_uniform_graph(num_node, self_link, neighbor)
        elif labeling_mode == 'distance*':
            A = tools.get_uniform_distance_graph(num_node, self_link, neighbor)
        elif labeling_mode == 'distance':
            A = tools.get_distance_graph(num_node, self_link, neighbor)
        elif labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        elif labeling_mode == 'DAD':
            A = tools.get_DAD_graph(num_node, self_link, neighbor)
        elif labeling_mode == 'DLD':
            A = tools.get_DLD_graph(num_node, self_link, neighbor)
        elif labeling_mode == 'FC':
            A= np.ones((num_node, num_node))
        # elif labeling_mode == 'customer_mode':
        #     pass
        else:
            raise ValueError()

        print(A)
        return A
        
class AdjMatrixGraph:
    def __init__(self, *args, **kwargs):
        self.edges = neighbor
        self.num_nodes = num_node
        self.self_loops = [(i, i) for i in range(self.num_nodes)]
        self.A_binary = tools.get_adjacency_matrix(self.edges, self.num_nodes)
        self.A_binary_with_I = tools.get_adjacency_matrix(self.edges + self.self_loops, self.num_nodes)
        self.A = tools.normalize_adjacency_matrix(self.A_binary)
        
class AdjMatrixGraph4lh:
    def __init__(self, *args, **kwargs):
        self.edges = neighbor1
        self.num_nodes = num_node1
        self.self_loops = [(i, i) for i in range(self.num_nodes)]
        self.A_binary = tools.get_adjacency_matrix(self.edges, self.num_nodes)
        self.A_binary_with_I = tools.get_adjacency_matrix(self.edges + self.self_loops, self.num_nodes)
        self.A = tools.normalize_adjacency_matrix(self.A_binary)
        
class AdjMatrixGraph4rh:
    def __init__(self, *args, **kwargs):
        self.edges = neighbor2
        self.num_nodes = num_node2
        self.self_loops = [(i, i) for i in range(self.num_nodes)]
        self.A_binary = tools.get_adjacency_matrix(self.edges, self.num_nodes)
        self.A_binary_with_I = tools.get_adjacency_matrix(self.edges + self.self_loops, self.num_nodes)
        self.A = tools.normalize_adjacency_matrix(self.A_binary)
        
class AdjMatrixGraph4ll:
    def __init__(self, *args, **kwargs):
        self.edges = neighbor3
        self.num_nodes = num_node3
        self.self_loops = [(i, i) for i in range(self.num_nodes)]
        self.A_binary = tools.get_adjacency_matrix(self.edges, self.num_nodes)
        self.A_binary_with_I = tools.get_adjacency_matrix(self.edges + self.self_loops, self.num_nodes)
        self.A = tools.normalize_adjacency_matrix(self.A_binary)
        
class AdjMatrixGraph4rl:
    def __init__(self, *args, **kwargs):
        self.edges = neighbor4
        self.num_nodes = num_node4
        self.self_loops = [(i, i) for i in range(self.num_nodes)]
        self.A_binary = tools.get_adjacency_matrix(self.edges, self.num_nodes)
        self.A_binary_with_I = tools.get_adjacency_matrix(self.edges + self.self_loops, self.num_nodes)
        self.A = tools.normalize_adjacency_matrix(self.A_binary)
        
class AdjMatrixGraph4torso:
    def __init__(self, *args, **kwargs):
        self.edges = neighbor5
        self.num_nodes = num_node5
        self.self_loops = [(i, i) for i in range(self.num_nodes)]
        self.A_binary = tools.get_adjacency_matrix(self.edges, self.num_nodes)
        self.A_binary_with_I = tools.get_adjacency_matrix(self.edges + self.self_loops, self.num_nodes)
        self.A = tools.normalize_adjacency_matrix(self.A_binary)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    graph = AdjMatrixGraph()
    A, A_binary, A_binary_with_I = graph.A, graph.A_binary, graph.A_binary_with_I
    f, ax = plt.subplots(1, 3)
    ax[0].imshow(A_binary_with_I, cmap='gray')
    ax[1].imshow(A_binary, cmap='gray')
    ax[2].imshow(A, cmap='gray')
    plt.show()
    print(A_binary_with_I.shape, A_binary.shape, A.shape)
