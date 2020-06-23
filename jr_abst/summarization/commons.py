#Commons
from jr_abst.summarization.graph import Graph

def build_graph(sequence):
    graph = Graph()
    for item in sequence:
        if not graph.has_node(item):
            graph.add_node(item)
    return graph


def remove_unreachable_nodes(graph):
    for node in graph.nodes():
        if all(graph.edge_weight((node, other)) == 0 for other in graph.neighbors(node)):
            graph.del_node(node)