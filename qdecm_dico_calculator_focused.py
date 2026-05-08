import os, sys
import platform
import numpy as np
import pickle 
import datetime as dt
from collections import defaultdict

from dcms.models import DCMModel, DECMModel, qDECMModel, DWCMModel

from auxiliary_functions import el2ks

# Set HOME directory based on the operating system
if platform.system() == 'Darwin':
    HOME = '/Users/fabio/Documents/Lavoro/PythonFiles/bowtie2_py310/bowtie2/'
elif platform.system() == 'Linux':
    HOME = '/home/sarawalk/bowtie2_py39/bowtie2/'
else:
    raise RuntimeError(f"Unsupported OS: {platform.system()}")

sys.path.insert(0, HOME)
DATA_FOLDER=HOME+'dati_elezioni/'
TEST_FOLDER=HOME+'tests/'

#DATASET='quirinale'
#DATASET='crisi'
DATASET='ita'
DICO=1
#DICO=0
#DICO=5

MAX_TIME_HOURS=6
#MAX_TIME_HOURS=2
TOL=1e-5
ANDERSON=10
HUB_TH=5
#GAMMA=1.2
GAMMA=0.
RECYCLE_TOPOLOGY=False
RECYCLE_WEIGHTS=True



def main():

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
    print(f'[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] {dataset_name.title()}')

    # Load the DiCo information
    dico=np.genfromtxt(DATA_FOLDER+dico_file, delimiter=',',skip_header=1, autostrip=True, dtype=[('user_id', '>U50'), ('dico', '>U2'), ('h_dico', 'U2'), ('i_dico', 'U2')])
    
    # Load the edge list
    el=np.genfromtxt(DATA_FOLDER+el_file, delimiter=',', skip_header=1,autostrip=True, dtype=[('source_id', '>U50'), ('target_id', '>U20'),('weight', 'i4')])

    # Select correct dicos
    dico_dict={}
    bad_dicos=[]
    for d in dico:
        if d['dico'].isnumeric():
            dico_dict[d['user_id']]=int(d['dico'])
        else:
            if d['dico'] not in bad_dicos:
                bad_dicos.append(d['dico'])

    
    # Nodes
    n_nodes=np.concatenate((el['source_id'], el['target_id']))
    n_nodes=np.unique(n_nodes)

    

    # Edges
    
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

    
    dicos=list(el_dico.keys())
    dicos.sort()
    
    print(f'[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] Processing DiCo class {dico_class}...')
    sys.stdout.flush()
    aux=el2ks(el_dico[dico_class])
    
    # consistency checks: topology
    assert aux[0].sum()==aux[1].sum()==len(el_dico[dico_class])
    # consistency checks: weights
    assert aux[2].sum()==aux[3].sum()

    print(f'[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] N(nodes)={len(aux[4]):,}, N(edges)={len(el_dico[dico_class]):,}, density={len(el_dico[dico_class])/len(aux[4])**2:.2e}')
    sys.stdout.flush()

    qdecm_filename=TEST_FOLDER+f'{dataset_name}_dico{dico_class}_qdecm.pkl'
    if os.path.exists(qdecm_filename):
        # check the existing solution
        with open(qdecm_filename, 'rb') as f:
            model=pickle.load(f)

        # tackle the existing solution if it did not converge
        if hasattr(model, 'sol') and hasattr(model.sol, 'converged') and model.sol.converged:
            print(f'[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] Existing QDECM solution for DiCo class {dico_class} is already converged, exiting...')
            return
        
        print(f'[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] qDECM, pytorch, theta (max: {MAX_TIME_HOURS:} hours)')
        qdecm=qDECMModel(aux[0], aux[1], aux[2], aux[3])

        # check if the solution for the topology can be used
        if RECYCLE_TOPOLOGY and hasattr(model, 'sol') and hasattr(model.sol, 'residuals_topo')  and model.sol.residuals_topo[-1]<TOL:
            ic_topo=model.sol.best_theta[:2*model.N]
        else:
            ic_topo="degrees"

        # if the solution for the weights did no converge, but it was quite close to convergence
        # we can still use the best theta found for the weights as an initial condition for the new run, which can help convergence
        if RECYCLE_WEIGHTS and hasattr(model, 'sol') and hasattr(model.sol, 'residuals_weights') and model.sol.residuals_weights[-1]<10**2*TOL:
            ic_wei=model.sol.best_theta[2*model.N:]
        else:
            ic_wei="topology"
    else:
        print(f'[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] qDECM, pytorch, theta (max: {MAX_TIME_HOURS:} hours)')
        qdecm=qDECMModel(aux[0], aux[1], aux[2], aux[3])
        ic_topo="degrees"
        ic_wei="topology"


    try:
        qdecm.solve_tool(tol=TOL, ic_topo=ic_topo, ic_wei=ic_wei, backend='pytorch', max_time=MAX_TIME_HOURS*3600, verbose=True, monitor=True, anderson_depth=ANDERSON, hub_sk_threshold=HUB_TH, backtracking_gamma=GAMMA)
    except Exception as e:
        print(f'[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] Error solving QDECM with pytorch and theta: {e}')
        sys.stdout.flush()

    with open(qdecm_filename, 'wb') as f:
        pickle.dump(qdecm, f)
    # elapsed time (in hours and minutes)
    t_ets=qdecm.sol.elapsed_time
    eth=t_ets//3600
    etm=(t_ets % 3600)/60

    if qdecm.sol.converged:
        convergence='converged'
    else:
        convergence='did not converge'
    print(f'[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] QDECM {convergence} in {int(eth):2d} h and {etm:2.2f} m, MRE(degrees)={qdecm.constraint_error_topology(qdecm.sol.best_theta[:2*qdecm.N]):.2e}, MRE(strengths)={qdecm.constraint_error_strength(qdecm.sol.best_theta[:2*qdecm.N], qdecm.sol.best_theta[2*qdecm.N:]):.2e} (peak RAM={qdecm.sol.peak_ram_bytes//1024**2} MB)')
    sys.stdout.flush()

if __name__ == "__main__":
    main()
