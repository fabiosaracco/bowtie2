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


def bic(model, decm_like=False):
    """
    Compute the Bayesian Information Criterion (BIC) for a given model.

    Parameters:
    model: one of dcms models
    decm_like: bool, whether to use the DEC-M like formulation
    Returns:
    float: The BIC value.
    """

    num_parameters = len(model.sol.best_theta)
    if decm_like:
        n_nodes = num_parameters/4  
    else:
        n_nodes = num_parameters/2
    n = n_nodes*(n_nodes-1)  # Number of observations

    # as in the original BIC formula, 
    # we use the negative log-likelihood and the number of parameters 
    # to compute the BIC value
    bic_value = 2 * model.neg_log_likelihood(model.sol.best_theta) + num_parameters * np.log(n)
    return bic_value