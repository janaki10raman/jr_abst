#graph
from abc import ABCMeta, abstractmethod

class IGraph(object):
    __metaclass__ = ABCMeta

    #abstractmethods
    def __len__(self):
        #Returns number of nodes in graph.
        pass

    def nodes(self):
        #Returns all nodes of graph.
        pass

    def edges(self):
        #Returns all edges of graph.
        pass

    def neighbors(self, node):
        #Return all nodes that are directly accessible from given node.
        pass

    def has_node(self, node):
        #Returns whether the requested node exists.
        pass

    def add_node(self, node):
        #Adds given node to the graph.
        pass

    def add_edge(self, edge, wt=1):
        #Adds an edge to the graph connecting two nodes. An edge, here is a tuple of two nodes.
        pass

    def has_edge(self, edge):
        #Returns whether an edge exists.
        pass

    def edge_weight(self, edge):
        #Returns weigth of given edge.
        pass

    def del_node(self, node):
        #Removes node and its edges from the graph.
        pass


class Graph(IGraph):
    #Implementing the undirected graph, based on IGraph.
    DEFAULT_WEIGHT = 0

    def __init__(self):
        self.node_neighbors = {}

    def __len__(self):
        #Returns number of nodes in graph.
        return len(self.node_neighbors)

    def has_edge(self, edge):
        #Returns whether an edge exists.
        u, v = edge
        return (u in self.node_neighbors
                and v in self.node_neighbors
                and v in self.node_neighbors[u]
                and u in self.node_neighbors[v])

    def edge_weight(self,edge):
        #Returns weight of given edge.
        u, v = edge
        return self.node_neighbors.get(u, {}).get(v, self.DEFAULT_WEIGHT)

    def neighbors(self,node):
        #Returns all nodes that are directly accessible from given node.
        return list(self.node_neighbors[node])

    def has_node(self,node):
        #Returns whether the requested node exists.
        return node in self.node_neighbors

    def add_edge(self, edge, wt=1):
        #Adds an edge to the graph connecting two nodes.
        if wt == 0.0:
            # empty edge is similar to no edge at all or removing it
            if self.has_edge(edge):
                self.del_edge(edge)
            return
        u, v = edge
        if v not in self.node_neighbors[u] and u not in self.node_neighbors[v]:
            self.node_neighbors[u][v] = wt
            if u != v:
                self.node_neighbors[v][u] = wt
        else:
            raise ValueError("Edge (%s, %s) already in graph" % (u, v))

    def add_node(self, node):
        #Adds given node to the graph.
        if node in self.node_neighbors:
            raise ValueError("Node %s already in graph" % node)
        self.node_neighbors[node] = {}

    def nodes(self):
        #Returns all nodes of the graph.
        return list(self.node_neighbors)

    def edges(self):
        #Returns all edges of the graph.
        return list(self.iter_edges())

    def iter_edges(self):
        #Returns iterator of all edges of the graph.
        for u in self.node_neighbors:
            for v in self.node_neighbors[u]:
                yield (u,v)

    def del_node(self, node):
        #Removes given node and its edges from the graph.
        for each in self.neighbors(node):
            if each != node:
                self.del_edge((each, node))
        del self.node_neighbors[node]

    def del_edge(self, edge):
        #Removes given edges from the graph.
        u, v = edge
        del self.node_neighbors[u][v]
        if u != v:
            del self.node_neighbors[v][u]