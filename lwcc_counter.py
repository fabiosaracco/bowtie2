import numpy as np
import datetime as dt
import igraph
from tqdm import trange, tqdm


def edges2lwcc(edge_list):
    g=igraph.Graph(directed=True)
    # get the node list
    nodes=np.unique(edge_list)
    g.add_vertices(nodes)
    g.add_edges(edge_list)
    # calculate the lwcc
    _wccs=g.components(mode='Weak')
    # get all WCCs
    l_wccs=np.array([len(_cl) for _cl in _wccs])
    # get the length of all WCCs    
    l_lwcc=np.max(l_wccs)
    return l_lwcc
    