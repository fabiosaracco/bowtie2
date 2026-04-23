import os, sys
import platform
import numpy as np
import pickle 
import datetime as dt
from collections import defaultdict
from dcms.models import DCMModel, DECMModel, ADECMModel, DWCMModel

if platform.system() == 'Darwin':
    HOME = '/Users/fabio/Documents/Lavoro/PythonFiles/bowtie2_py310/bowtie2/'
elif platform.system() == 'Linux':
    HOME = '/home/sarawalk/bowtie2_py39/bowtie2/'
else:
    raise RuntimeError(f"Unsupported OS: {platform.system()}")

sys.path.insert(0, HOME)
DATA_FOLDER=HOME+'dati_elezioni/'

def main():
    files=os.listdir(DATA_FOLDER)
    files.sort()
    l_dataset=len(files)//2

    # select a dataset
    i_file=0

    # get edgelist and dico
    el=np.genfromtxt(
        DATA_FOLDER+files[i_file+1],
        delimiter=',',
        skip_header=1,
        autostrip=True,
        dtype=[('source_id', '>U50'), ('target_id', '>U20'),('weight', 'i4')]
    )

    dico=np.genfromtxt(
    DATA_FOLDER+files[i_file],
    delimiter=',',
    skip_header=1,
    autostrip=True,
    dtype=[('user_id', '>U50'), ('dico', '>U2'), ('h_dico', 'U2'), ('i_dico', 'U2')]
    )

    dico_dict={}
    bad_dicos=[]
    for d in dico:
        if d['dico'].isnumeric():
            dico_dict[d['user_id']]=int(d['dico'])
        #else:
        #    if d['dico'] not in bad_dicos:
        #        bad_dicos.append(d['dico'])
        #        print(d['dico'])


    
    # Raggruppo in liste per efficienza, poi converto in array strutturati come el
    _tmp = defaultdict(list)
    for edge in el:
        src = edge['source_id'].strip()
        tgt = edge['target_id'].strip()
        d_src = dico_dict.get(src)
        if d_src is not None and d_src == dico_dict.get(tgt):
            _tmp[d_src].append(edge)


    el_dico = defaultdict(
    lambda: np.empty(0, dtype=el.dtype),
    {k: np.array(v, dtype=el.dtype) for k, v in _tmp.items()}
    )


    #for key in el_dico.keys():
    #    print(key, len(el_dico[key]))


    del _tmp



    aux=el2ks(el_dico[1])# DiCo 1

    assert aux[0].sum()==aux[1].sum()==len(el_dico[1])

    assert aux[2].sum()==aux[3].sum()

    # Number of nodes, Number of edges, edge density
    print(f'[{dt.datetime.now():%H:%M:%S}] Number of nodes: {len(aux[4])}, Number of edges: {len(el_dico[1])}, Edge density: {len(el_dico[1])/len(aux[4])**2:.2e}')
    print(f'[{dt.datetime.now():%H:%M:%S}] ---DiCo 1---')
    # ### aDECM

    
    # #### Pytorch, $\theta$
    print(f'[{dt.datetime.now():%H:%M:%S}] aDECM, pytorch, theta')
    if not os.path.exists(HOME+f'/test/crisis_adecm_old_theta.pkl'):
        adecm_old=ADECMModel(aux[0], aux[1], aux[2], aux[3])
        try:
            adecm_old.solve_tool(tol=1e-5, max_iter=10000, backend='pytorch', verbose=True)
            # with backend='pytorch'
            with open(HOME+f'tests/crisis_adecm_old_theta.pkl', 'wb') as f:
                pickle.dump(adecm_old, f)
        except Exception as e:
            print(f'Error solving aDECM with pytorch and theta: {e}')
    
    
    # #### Numba, $\theta$, n_procs=8
    print(f'[{dt.datetime.now():%H:%M:%S}] ADECM, numba, theta, n_procs=8')
    adecm=ADECMModel(aux[0], aux[1], aux[2], aux[3])
    nprocs=8
    try:
        adecm.solve_tool(tol=1e-5, max_iter=10000, num_threads=nprocs, backend='numba', verbose=True)
        # with backend='auto' (default), that is numba for N>5k
        with open(HOME+f'tests/crisis_adecm_new_theta_nprocs_{nprocs}.pkl', 'wb') as f:
            pickle.dump(adecm, f)
    except Exception as e:
        print(f'Error solving aDECM with numba and theta: {e}')


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

if __name__ == "__main__":
    main()
