#pagerank_weighted
import numpy
from numpy import empty as empty_matrix
from scipy.linalg import eig
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
from six.moves import range
from jr_abst.utils import deprecated


def pagerank_weighted(graph, damping=0.85):
    #Get dictionary of graph nodes and its ranks.
    coeff_adjacency_matrix = build_adjacency_matrix(graph, coeff=damping)
    probabilities = (1 - damping) / float(len(graph))
    pagerank_matrix = coeff_adjacency_matrix.toarray()
    # trying to minimize memory allocations
    pagerank_matrix += probabilities
    vec = principal_eigenvector(pagerank_matrix.T)
    # Because pagerank_matrix is positive, vec is always real (i.e. not complex)
    return process_results(graph, vec.real)


def build_adjacency_matrix(graph, coeff=1):
    #Get matrix representation of given graph.
    row = []
    col = []
    data = []
    nodes = graph.nodes()
    nodes2id = {v: i for i, v in enumerate(nodes)}
    length = len(nodes)
    for i in range(length):
        current_node = nodes[i]
        neighbors = graph.neighbors(current_node)
        neighbors_sum = sum(graph.edge_weight((current_node, neighbor)) for neighbor in neighbors)
        for neighbor in neighbors:
            edge_weight = float(graph.edge_weight((current_node, neighbor)))
            if edge_weight != 0.0:
                row.append(i)
                col.append(nodes2id[neighbor])
                data.append(coeff * edge_weight / neighbors_sum)
    return csr_matrix((data, (row, col)), shape=(length, length))


def build_probability_matrix(graph, coeff=1.0):
    #Get square matrix of shape nxn, where n is number of nodes of the given graph.
    dimension = len(graph)
    matrix = empty_matrix((dimension, dimension))
    probability = coeff / float(dimension)
    matrix.fill(probability)
    return matrix


def principal_eigenvector(a):
    #Get eigenvector of square matrix a.
    # Note that we prefer to use eigs even for dense matrix
    # because we need only one eigenvector.
    if len(a) < 3: #works only for dim A < 3
        vals, vecs = eig(a)
        ind = numpy.abs(vals).argmax()
        return vecs[:, ind]
    else:
        vals, vecs = eigs(a, k=1)
        return vecs[:, 0]


def process_results(graph, vec):
    #Get graph nodes and corresponding absolute values of provided eigenvector.
    scores = {}
    for i, node in enumerate(graph.nodes()):
        scores[node] = abs(vec[i])
    return scores