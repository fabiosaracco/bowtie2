import os, sys, pickle
import platform
import datetime as dt
import numpy as np
from collections import defaultdict

from sam_bowtie import validate

if platform.system() == 'Darwin':
    print(f'[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] Air!')
    HOME = '/Users/fabio/Documents/Lavoro/PythonFiles/bowtie2_py310/bowtie2/'
elif platform.system() == 'Linux':
    print(f'[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] Stella!')
    HOME = '/home/sarawalk/bowtie2_py39/bowtie2/'
else:
    raise RuntimeError(f"Unsupported OS: {platform.system()}")

sys.path.insert(0, HOME)

DATA_FOLDER=HOME+'dati_elezioni/'
TEST_FOLDER=HOME+'tests/'
PVALUE_FOLDER=HOME+'pvalues/'

#N_RUNS=2*1000
N_RUNS=5*1000

def get_input_data():
    dataset_keyword=DATASET
    dico_class=DICO
        
    files=os.listdir(DATA_FOLDER)
    files.sort()
    
    # Each dataset has two files:
    # - ???_dicos.csv: DiCo information per node
    # - ???_weighted_edgelist.csv: edge list with columns source_id, target_id, weight
    
    # Focus on a single dataset
    files=[f for f in files if f.startswith(dataset_keyword)]
    
    dico_file = files[0]
    el_file = files[1]
    dataset_name=dico_file[:-10]
    print(f'[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] ***{dataset_name.title()}***')

    # Load the DiCo information
    dico=np.genfromtxt(DATA_FOLDER+dico_file, delimiter=',',skip_header=1, autostrip=True, dtype=[('user_id', '>U50'), ('dico', '>U2'), ('h_dico', 'U2'), ('i_dico', 'U2')])
        
    # Load the edge list
    el=np.genfromtxt(DATA_FOLDER+el_file, delimiter=',', skip_header=1,autostrip=True, dtype=[('source_id', '>U50'), ('target_id', '>U20'),('weight', 'i4')])

    # Select correct dicos
    print(f'[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] Selecting only interpretable dicos...')
    sys.stdout.flush()
    dico_dict={}
    bad_dicos=[]
    for d in dico:
        if d['dico'].isnumeric():
            dico_dict[d['user_id']]=int(d['dico'])
        else:
            if d['dico'] not in bad_dicos:
                bad_dicos.append(d['dico'])

    cacca=np.unique(list(dico_dict.values()), return_counts=True)
    #print(f'[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] DiCo nodes distribution:') 
    #for entry in np.vstack(cacca).T:
    #    print(f'{entry[0]}, {entry[1]:7,d}')
    #sys.stdout.flush()

    # Nodes
    n_nodes=np.concatenate((el['source_id'], el['target_id']))
    n_nodes=np.unique(n_nodes)

    #print(f'[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] N(nodes)={len(n_nodes):,}, N(nodes in dico)={len(dico_dict):,}, share={len(dico_dict)/len(n_nodes):.3f}')

    # Edges
    n_edges = len(el)
    _tmp = defaultdict(list)
    # auxiliary defaultdict to group edges by dico class
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

    del _tmp

    cacca=np.array([[key, len(el_dico[key])] for key in el_dico.keys()])

    #print(f'[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] DiCo edges distribution:') 
    dicos=list(el_dico.keys())
    dicos.sort()
    
    #for d in dicos:
    #    print(f'{d}, {len(el_dico[d]):8,d}')
    #sys.stdout.flush()
    
    print(f'[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] Processing DiCo class {dico_class}...')
    sys.stdout.flush()
    aux=el2ks(el_dico[dico_class])

    # consistency checks: topology
    assert aux[0].sum()==aux[1].sum()==len(el_dico[dico_class])
    # consistency checks: weights
    assert aux[2].sum()==aux[3].sum()

    return aux, n_edges, dataset_name

def main():

    aux, n_edges, dataset_name=get_input_data()
        
    
    qdecm_filename=TEST_FOLDER+f'{dataset_name}_dico{d}_qdecm.pkl'
    pvalue_block_filename=PVALUE_FOLDER+f'{dataset_name}_dico{d}_pvalues_blocks.pkl'
    pvalue_flux_filename=PVALUE_FOLDER+f'{dataset_name}_dico{d}_pvalues_fluxes.pkl'
    #if os.path.exists(pvalue_block_filename) and os.path.exists(pvalue_flux_filename):
    #    print(f'[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] P-value files for DiCo {d} already exist, skipping...')
    #    continue
    counter=0
    while os.path.exists(pvalue_block_filename):
        pvalue_block_filename=PVALUE_FOLDER+f'{dataset_name}_dico{d}_pvalues_blocks_{counter}.pkl'
        pvalue_flux_filename=PVALUE_FOLDER+f'{dataset_name}_dico{d}_pvalues_fluxes_{counter}.pkl'
        counter+=1
        
    
    if os.path.exists(qdecm_filename):
        # check if the file was created/modified today
        #file_mtime = dt.date.fromtimestamp(os.path.getmtime(qdecm_filename))
        #if file_mtime == dt.date.today():
        with open(qdecm_filename, 'rb') as f:
            qdecm=pickle.load(f)
        if hasattr(qdecm, 'sol') and qdecm.sol.converged:
            print(f'[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] Processing DiCo {d}...')
            sys.stdout.flush()
            block_dict, flux_dict=validate(el_dico[d], qdecm, n_runs=N_RUNS, verbose=True)
            with open(pvalue_block_filename, 'wb') as f:
                pickle.dump(block_dict, f)
            with open(pvalue_flux_filename, 'wb') as f:
                pickle.dump(flux_dict, f)

if __name__ == "__main__":
    main()
