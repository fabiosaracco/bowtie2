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

DATASET='ita_elections'
#DATASET='crisi'
#DICO=0
#DICO=1
DICO=2
#DICO=3


MAX_TIME_HOURS=24
MAX_ITER=100000
TOL=1e-5
ANDERSON=10
HUB_TH=5
GAMMA=0.
MONITOR=False
RECYCLE_SOL=True
BLOWUP=None

def jackie_chunk(model, n_chunks, ic, filename_checkpoint):
    '''
    Solve the model in chunks, allowing for a restart if the process is interrupted.
    This function will save the best solution found so far after each chunk.'''
    global_best_theta, best_mre = None, float("inf")
    theta = None  # None => solver picks the default ic on the first chunk
    max_time_chunk=MAX_TIME_HOURS * 3600 // n_chunks  # divide total time by number of chunks
    max_iter_chunk=MAX_ITER // n_chunks  # divide total iterations by number of chunks
    
    for chunk in range(n_chunks):
        print(f'[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] Starting chunk {chunk+1}/{n_chunks}...')
        if chunk==0:
            model.solve_tool(tol=TOL, backend='pytorch', ic=ic, max_time=max_time_chunk, max_iter=max_iter_chunk, verbose=True, monitor=MONITOR, anderson_depth=ANDERSON, hub_sk_threshold=HUB_TH, backtracking_gamma=GAMMA, multi_start=False, blowup_factor=BLOWUP)
        else:
            model.solve_tool(tol=TOL, backend='pytorch', ic=global_best_theta, max_time=max_time_chunk, max_iter=max_iter_chunk, verbose=True, monitor=MONITOR, anderson_depth=ANDERSON, hub_sk_threshold=HUB_TH, backtracking_gamma=GAMMA, multi_start=False, blowup_factor=BLOWUP)
        if model.sol.mre < best_mre:
            best_mre = model.sol.mre
            global_best_theta = model.sol.best_theta.copy()
        theta = model.sol.theta          # continue from where this chunk left off
        save_checkpoint(chunk, theta, global_best_theta, best_mre, filename_checkpoint)  # survive a restart
        if model.sol.converged:
            break
    return model

def save_checkpoint(chunk, theta, global_best_theta, best_mre, filename):
    checkpoint = {
        'chunk': chunk,
        'theta': theta,
        'global_best_theta': global_best_theta,
        'best_mre': best_mre
    }
    with open(filename, 'ab') as f:
        pickle.dump(checkpoint, f)

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

    print(f'[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] N(nodes)={len(aux[4]):,}, N(edges)={n_edges:,}, density={n_edges/len(aux[4])**2:.2e}')
    sys.stdout.flush()

    
    dico_class=DICO

    filename=HOME+f'/tests/{dataset_name}_dico{dico_class}_decm.pkl'
    
    if os.path.exists(filename):
        # check if the file was created/modified today
        #file_mtime = dt.date.fromtimestamp(os.path.getmtime(filename))
        #if file_mtime == dt.date.today():
        
        with open(filename, 'rb') as f:
            old_decm=pickle.load(f)
        if hasattr(old_decm, 'sol') and hasattr(old_decm.sol, 'converged') and old_decm.sol.converged and old_decm.sol.mre<TOL:
            print(f'[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] A converged solution for decm with pytorch and theta already exists. Skipping...')
            sys.stdout.flush()
            return


    # check if a checkpoint exists and can be used as a starting point
    filename_checkpoint = filename.replace('tests', 'checkpoints')
    if RECYCLE_SOL and os.path.exists(filename_checkpoint):
        with open(filename_checkpoint, 'rb') as f:
            try:
                while True:
                    checkpoint = pickle.load(f)
            except EOFError:
                pass  # reached the end of the file
        if checkpoint['best_mre'] < TOL:
            print(f'[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] A converged solution for decm with pytorch and theta already exists in the checkpoint. Skipping...')
            sys.stdout.flush()
            return
        else:
            print(f'[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] Recycling existing non-converged solution from checkpoint...')
            ic=checkpoint['global_best_theta']
    else:
        ic="degrees"
        
    print(f'[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] DECM, pytorch, theta (max: {MAX_TIME_HOURS:} hours)')
    decm=DECMModel(aux[0], aux[1], aux[2], aux[3])

    _start=dt.datetime.now()
    decm=jackie_chunk(decm, n_chunks=100, ic=ic, filename_checkpoint=filename_checkpoint)
    _end=dt.datetime.now()
    elapsed_time = (_end - _start).total_seconds()
    
    
    eth=elapsed_time//3600
    etm=(elapsed_time % 3600)/60

    if decm.sol.converged:
        print(f'[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] DECM converged in {int(eth):2d} h and {etm:2.2f} m, MRE={decm.sol.mre:.4e}, (peak RAM={decm.sol.peak_ram_bytes//1024**2} MB)')
            
    else:
        print(f'[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] DECM did not converge in {int(eth):2d} h and {etm:2.2f} m, MRE={decm.sol.mre:.4e}, (peak RAM={decm.sol.peak_ram_bytes//1024**2} MB)')
    sys.stdout.flush()

    final_filename=filename
    counter=0
    if os.path.exists(filename):
        final_filename=filename.replace('.pkl', f'_{counter}.pkl')
    while os.path.exists(final_filename):
        counter+=1
        final_filename=filename.replace('.pkl', f'_{counter}.pkl')
    
    print(f'[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] file name: {final_filename}')
    sys.stdout.flush()
    with open(final_filename, 'wb') as f:
        pickle.dump(decm, f)
    
    print(f'[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] Saved!')
    sys.stdout.flush()
    


if __name__ == "__main__":
    main()
    if platform.system() == 'Darwin':
        os.system("afplay ta-da.mp3")
