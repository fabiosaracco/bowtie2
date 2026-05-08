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
        print(f'[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] ***{dataset_name.title()}***')

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

        dicos=list(el_dico.keys())
        dicos.sort()
        
        for d in dicos:
            qdecm_filename=TEST_FOLDER+f'{dataset_name}_dico{d}_qdecm.pkl'
            pvalue_block_filename=PVALUE_FOLDER+f'{dataset_name}_dico{d}_pvalues_blocks.pkl'
            pvalue_flux_filename=PVALUE_FOLDER+f'{dataset_name}_dico{d}_pvalues_fluxes.pkl'
            if os.path.exists(pvalue_block_filename) and os.path.exists(pvalue_flux_filename):
                print(f'[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] P-value files for DiCo {d} already exist, skipping...')
                continue
            if os.path.exists(qdecm_filename):
                # check if the file was created/modified today
                #file_mtime = dt.date.fromtimestamp(os.path.getmtime(qdecm_filename))
                #if file_mtime == dt.date.today():
                with open(qdecm_filename, 'rb') as f:
                    qdecm=pickle.load(f)
                if hasattr(qdecm, 'sol') and qdecm.sol.converged:
                    print(f'[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] Processing DiCo {d} with {len(el_dico[d]):,} edges...')
                    sys.stdout.flush()
                    block_dict, flux_dict=validate(el_dico[d], qdecm, n_runs=2*1000, verbose=True)
                    with open(pvalue_block_filename, 'wb') as f:
                        pickle.dump(block_dict, f)
                    with open(pvalue_flux_filename, 'wb') as f:
                        pickle.dump(flux_dict, f)

if __name__ == "__main__":
    main()
