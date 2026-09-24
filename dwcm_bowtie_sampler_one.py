"""Process a single (dataset, dico) network's bowtie p-value sampling.

Run as its own fresh process (invoked as a subprocess by
dwcm_bowtie_sampler.py, one call per network) so that ProcessPoolExecutor
inside sam_bowtie.validate() always forks from a clean, single-threaded
parent -- a long-lived driver process that creates a NEW pool for every
network hit a macOS/Apple-Silicon fork-safety crash on the second pool
creation (native library thread-pool state left over from the first
validate() call corrupts the next fork()). One-process-per-network sidesteps
that entirely.
"""
import argparse
import os, sys, pickle
import platform
import datetime as dt
import numpy as np
from collections import defaultdict

from sam_bowtie import validate

if platform.system() == 'Darwin':
    HOME = '/Users/fabio/Documents/Lavoro/PythonFiles/bowtie2_py310/bowtie2/'
elif platform.system() == 'Linux':
    HOME = '/home/sarawalk/bowtie2_py39/bowtie2/'
else:
    raise RuntimeError(f"Unsupported OS: {platform.system()}")

sys.path.insert(0, HOME)

DATA_FOLDER = HOME + 'dati_elezioni/'
TEST_FOLDER = HOME + 'tests/'
PVALUE_FOLDER = HOME + 'pvalues/'

N_RUNS = 5 * 1000


def log(msg):
    print(f'[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--dico', required=True, type=int)
    args = ap.parse_args()
    dataset_name, d = args.dataset, args.dico

    dico = np.genfromtxt(
        DATA_FOLDER + f'{dataset_name}_dicos.csv', delimiter=',', skip_header=1,
        autostrip=True, encoding='latin-1',
        dtype=[('user_id', '>U50'), ('dico', '>U2'), ('h_dico', 'U2'), ('i_dico', 'U2')],
    )
    el = np.genfromtxt(
        DATA_FOLDER + f'{dataset_name}_weighted_edgelist.csv', delimiter=',', skip_header=1,
        autostrip=True, encoding='latin-1',
        dtype=[('source_id', '>U50'), ('target_id', '>U20'), ('weight', 'i4')],
    )

    dico_dict = {row['user_id']: int(row['dico']) for row in dico if row['dico'].isnumeric()}

    edges = []
    for edge in el:
        src = edge['source_id'].strip()
        tgt = edge['target_id'].strip()
        d_src = dico_dict.get(src)
        if d_src is not None and d_src == d and d_src == dico_dict.get(tgt):
            edges.append(edge)
    el_d = np.array(edges, dtype=el.dtype)

    dwcm_filename = TEST_FOLDER + f'{dataset_name}_dico{d}_dwcm_final.pkl'
    pvalue_block_filename = PVALUE_FOLDER + f'{dataset_name}_dico{d}_dwcm_pvalues_blocks.pkl'
    pvalue_flux_filename = PVALUE_FOLDER + f'{dataset_name}_dico{d}_dwcm_pvalues_fluxes.pkl'
    counter = 0
    while os.path.exists(pvalue_block_filename):
        pvalue_block_filename = PVALUE_FOLDER + f'{dataset_name}_dico{d}_dwcm_pvalues_blocks_{counter}.pkl'
        pvalue_flux_filename = PVALUE_FOLDER + f'{dataset_name}_dico{d}_dwcm_pvalues_fluxes_{counter}.pkl'
        counter += 1

    if not os.path.exists(dwcm_filename):
        log(f'{dataset_name}_dico{d}: no DWCM checkpoint found, skipping.')
        return
    with open(dwcm_filename, 'rb') as f:
        dwcm = pickle.load(f)
    if not (hasattr(dwcm, 'sol') and dwcm.sol.converged):
        log(f'{dataset_name}_dico{d}: DWCM not converged, skipping.')
        return

    log(f'{dataset_name}_dico{d}: sampling ({N_RUNS} runs)...')
    block_dict, flux_dict = validate(el_d, dwcm, n_runs=N_RUNS, verbose=True)
    with open(pvalue_block_filename, 'wb') as f:
        pickle.dump(block_dict, f)
    with open(pvalue_flux_filename, 'wb') as f:
        pickle.dump(flux_dict, f)
    log(f'{dataset_name}_dico{d}: saved.')


if __name__ == "__main__":
    main()
