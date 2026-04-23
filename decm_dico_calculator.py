import os, sys
import platform
import numpy as np
import pickle 
import datetime as dt
from collections import defaultdict

from dcms.models import DCMModel, DECMModel, ADECMModel, DWCMModel

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

MAX_TIME_HOURS=6

def main():
    files=os.listdir(DATA_FOLDER)
    files.sort()

    # Each dataset has two files:
    # - ???_dicos.csv: DiCo information per node
    # - ???_weighted_edgelist.csv: edge list with columns source_id, target_id, weight

    for i in range(len(files)//2):
        dico_file = files[2*i]
        el_file = files[2*i + 1]
        dataset_name=dico_file[:-10]
        print(f'[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] {dataset_name.title()}')

        # Load the DiCo information
        dico=np.genfromtxt(DATA_FOLDER+dico_file, delimiter=',',skip_header=1, autostrip=True, dtype=[('user_id', '>U50'), ('dico', '>U2'), ('h_dico', 'U2'), ('i_dico', 'U2')])
        
        # Load the edge list
        el=np.genfromtxt(DATA_FOLDER+el_file, delimiter=',', skip_header=1,autostrip=True, dtype=[('source_id', '>U50'), ('target_id', '>U20'),('weight', 'i4')])

        # Select correct dicos
        print(f'[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] Selecting only interpretable dicos...')
        dico_dict={}
        bad_dicos=[]
        for d in dico:
            if d['dico'].isnumeric():
                dico_dict[d['user_id']]=int(d['dico'])
            else:
                if d['dico'] not in bad_dicos:
                    bad_dicos.append(d['dico'])

        cacca=np.unique(list(dico_dict.values()), return_counts=True)
        print(f'[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] DiCo nodes distribution:') 
        for entry in np.vstack(cacca).T:
            print(f'{entry[0]}, {entry[1]:7,d}')

        # Nodes
        n_nodes=np.concatenate((el['source_id'], el['target_id']))
        n_nodes=np.unique(n_nodes)

        print(f'[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] N(nodes)={len(n_nodes):,}, N(nodes in dico)={len(dico_dict):,}, share={len(dico_dict)/len(n_nodes):.3f}')

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
    
        print(f'[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] DiCo edges distribution:') 
        dicos=list(el_dico.keys())
        dicos.sort()
        
        for d in dicos:
            print(f'{d}, {len(el_dico[d]):8,d}')


        
        for dico_class in dicos:
            print(f'[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] Processing DiCo class {dico_class}...')
            aux=el2ks(el_dico[dico_class])
            # consistency checks: topology
            assert aux[0].sum()==aux[1].sum()==len(el_dico[dico_class])
            # consistency checks: weights
            assert aux[2].sum()==aux[3].sum()

            print(f'[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] N(nodes)={len(aux[4]):,}, N(edges)={len(el_dico[dico_class]):,}, density={len(el_dico[dico_class])/len(aux[4])**2:.2e}')

            print(f'[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] DECM, pytorch, theta (max: {MAX_TIME_HOURS:} hours)')

            decm=DECMModel(aux[0], aux[1], aux[2], aux[3])

            try:
                decm.solve_tool(tol=1e-6, backend='pytorch', max_time=MAX_TIME_HOURS*3600)
            except Exception as e:
                print(f'[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] Error solving DECM with pytorch and theta: {e}')
            # with backend='pytorch'
            with open(HOME+f'/test/{dataset_name}_dico{dico_class}_decm.pkl', 'wb') as f:
                pickle.dump(decm, f)
                
            # elapsed time (in hours and minutes)
            eth=decm.sol.elapsed_time//3600
            etm=(decm.sol.elapsed_time % 3600)//60
            
            if decm.converged:
                print(f'[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] DECM converged in {eth:2d} h and {etm:2d} m, MRE={decm.max_relative_error(decm.sol.theta):.2e} (peak RAM={decm.sol.peak_ram_bytes//1024**2} MB)')
            else:
                print(f'[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] DECM did not converge in {eth:2d} h and {etm:2d} m, MRE={decm.max_relative_error(decm.sol.theta):.2e} (peak RAM={decm.sol.peak_ram_bytes//1024**2} MB)')
                print(f'[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] Trying aDECM...')
                print(f'[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] aDECM, pytorch, theta (max: {MAX_TIME_HOURS:} hours)')
                adecm=ADECMModel(aux[0], aux[1], aux[2], aux[3])

                try:
                    adecm.solve_tool(tol=1e-6, backend='pytorch', max_time=MAX_TIME_HOURS*3600)
                except Exception as e:
                    print(f'[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] Error solving ADECM with pytorch and theta: {e}')
                # with backend='pytorch'
                with open(HOME+f'/test/{dataset_name}_dico{dico_class}_adecm.pkl', 'wb') as f:
                    pickle.dump(adecm, f)
                # elapsed time (in hours and minutes)
                t_ets=adecm.sol_topo.elapsed_time+adecm.sol_weights.elapsed_time
                eth=t_ets//3600
                etm=(t_ets % 3600)//60
            
                if adecm.sol_topo.converged and adecm.sol_weights.converged:
                    print(f'[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] ADECM converged in {eth:2d} h and {etm:2d} m, MRE(degrees)={adecm.constraint_error_topology(adecm.sol_topo.theta):.2e}, MRE(strengths)={adecm.constraint_error_strength(adecm.sol_topo.theta, adecm.sol_weights.theta):.2e} (peak RAM={adecm.sol_topo.peak_ram_bytes//1024**2} MB (topo), {adecm.sol_weights.peak_ram_bytes//1024**2} MB (weights))')
                else:
                    print(f'[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] ADECM did not converge in {eth:2d} h and {etm:2d} m, MRE(degrees)={adecm.constraint_error_topology(adecm.sol_topo.theta):.2e}, MRE(strengths)={adecm.constraint_error_strength(adecm.sol_topo.theta, adecm.sol_weights.theta):.2e} (peak RAM={adecm.sol_topo.peak_ram_bytes//1024**2} MB (topo), {adecm.sol_weights.peak_ram_bytes//1024**2} MB (weights))')

    



if __name__ == "__main__":
    main()