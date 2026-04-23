import os, sys
import numpy as np

def el2ks(el):
    all_nodes=np.concatenate((el['source_id'], el['target_id']))
    all_nodes=np.unique(all_nodes)
    k_out=np.zeros(len(all_nodes), dtype=int)
    k_in=np.zeros(len(all_nodes), dtype=int)
    s_out=np.zeros(len(all_nodes), dtype=int)
    s_in=np.zeros(len(all_nodes), dtype=int)
    node_index={node:i for i,node in enumerate(all_nodes)}
    for s,t,w in el:
        i_s=node_index[s]
        i_t=node_index[t]
        k_out[i_s]+=1
        k_in[i_t]+=1
        s_out[i_s]+=w
        s_in[i_t]+=w
    return k_out, k_in, s_out, s_in, all_nodes