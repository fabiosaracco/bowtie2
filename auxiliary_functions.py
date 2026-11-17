import os, platform, sys
import numpy as np
from collections import defaultdict

if platform.system() == 'Darwin':
    print('Air!')
    HOME = '/Users/fabio/Documents/Lavoro/PythonFiles/bowtie2_py310/bowtie2/'
elif platform.system() == 'Linux':
    print('Stella!')
    HOME = '/home/sarawalk/bowtie2_py39/bowtie2/'
else:
    raise RuntimeError(f"Unsupported OS: {platform.system()}")

sys.path.insert(0, HOME)
DATA_FOLDER=HOME+'dati_elezioni/'

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

def load_dico_edges(dataset, dico_class):
    files = [f for f in sorted(os.listdir(DATA_FOLDER)) if f.startswith(dataset)]
    dico_file, el_file = files[0], files[1]

    dico = np.genfromtxt(DATA_FOLDER + dico_file, delimiter=',', skip_header=1,
                          autostrip=True,
                          dtype=[('user_id', '>U50'), ('dico', '>U2'),
                                 ('h_dico', 'U2'), ('i_dico', 'U2')])
    el = np.genfromtxt(DATA_FOLDER + el_file, delimiter=',', skip_header=1,
                        autostrip=True,
                        dtype=[('source_id', '>U50'), ('target_id', '>U20'),
                               ('weight', 'i4')])

    dico_dict = {}
    for d in dico:
        if d['dico'].isnumeric():
            dico_dict[d['user_id']] = int(d['dico'])

    _tmp = defaultdict(list)
    for edge in el:
        src = edge['source_id'].strip()
        tgt = edge['target_id'].strip()
        d_src = dico_dict.get(src)
        if d_src is not None and d_src == dico_dict.get(tgt):
            _tmp[d_src].append(edge)

    if dico_class not in _tmp:
        raise ValueError(f'dico class {dico_class} not found in {dataset}')

    return np.array(_tmp[dico_class], dtype=el.dtype)