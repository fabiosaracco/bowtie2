"""Single-network DECM solve, driven by CLI args, for the 2026-08-28 batch.

Meant to be launched as a subprocess by run_decm_batch2608_stella.py, one
process per (dataset, dico) target, pinned to a single free core (via
taskset) with all BLAS/threading env vars forced to 1 thread by the caller.

Uses the validated recipe from the ita_elections_dico3 investigation
(z_clamp=1e-6 tied, noise_base=2e-3) as the default for every network in
this batch, since it was already swept and found to generalize (dcms test
matrices, power-law generators up to N=50k, crisi_dico2/3/4).
"""
import argparse
import os
import sys
import platform
import pickle
import datetime as dt
from collections import defaultdict

import numpy as np

from dcms.models import DECMModel
from auxiliary_functions import el2ks

if platform.system() == 'Darwin':
    HOME = '/Users/fabio/Documents/Lavoro/PythonFiles/bowtie2_py310/bowtie2/'
elif platform.system() == 'Linux':
    HOME = '/home/sarawalk/bowtie2_py39/bowtie2/'
else:
    raise RuntimeError(f"Unsupported OS: {platform.system()}")

sys.path.insert(0, HOME)
DATA_FOLDER = HOME + 'dati_elezioni/'
TEST_FOLDER = HOME + 'tests/'
LOG_FOLDER = HOME + 'logs/'

MAX_TIME_HOURS = 24
# Deliberately generous: for small/cheap networks the real budget should be
# max_time (24h), not an iteration cap -- a cheap network stuck in a
# stagnation plateau needs many patience-triggered restart cycles (every
# 750 iters, escalating to a noisy perturbed restart every 2nd stall) to
# have a fair shot at escaping, and at a few ms/iter it can run hundreds of
# thousands of iterations well within 24h. Large networks will hit max_time
# long before this iteration count regardless.
MAX_ITER = 300000
TOL = 1e-5
Z_CLAMP = 1e-6
NOISE_BASE = 2e-3
HUB_TH = 5.0
GAMMA = 0.0
ANDERSON = 10


def log(msg):
    print(f'[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}')
    sys.stdout.flush()


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--dico', required=True, type=int)
    args = ap.parse_args()

    dataset, dico_class = args.dataset, args.dico
    tag = f'{dataset}_dico{dico_class}'

    ckpt_path = TEST_FOLDER + f'batch2608_{tag}_decm.pkl'
    if os.path.exists(ckpt_path):
        with open(ckpt_path, 'rb') as f:
            old = pickle.load(f)
        if hasattr(old, 'sol') and getattr(old.sol, 'converged', False) and old.sol.mre < TOL:
            log(f'{tag}: already converged in a previous batch run, skipping.')
            return

    log(f'*** {tag} ***')
    el_dico = load_dico_edges(dataset, dico_class)
    k_out, k_in, s_out, s_in, nodes = el2ks(el_dico)

    assert k_out.sum() == k_in.sum() == len(el_dico)
    assert s_out.sum() == s_in.sum()

    log(f'N(nodes)={len(nodes):,}, N(edges)={len(el_dico):,}, '
        f'density={len(el_dico)/len(nodes)**2:.2e}')

    decm = DECMModel(k_out, k_in, s_out, s_in)

    log(f'DECM, backend=auto, num_threads=1, z_clamp={Z_CLAMP:.0e}, '
        f'noise_base={NOISE_BASE:.0e} (max: {MAX_TIME_HOURS} hours)')

    try:
        decm.solve_tool(
            ic='degrees',
            tol=TOL,
            max_iter=MAX_ITER,
            max_time=MAX_TIME_HOURS * 3600,
            anderson_depth=ANDERSON,
            multi_start=True,
            backend='auto',
            num_threads=1,
            verbose=True,
            monitor=False,
            hub_sk_threshold=HUB_TH,
            backtracking_gamma=GAMMA,
            z_clamp=Z_CLAMP,
            reduce_degeneracy=True,
            noise_base=NOISE_BASE,
        )

        t = decm.sol.elapsed_time
        eh, em = int(t // 3600), (t % 3600) / 60

        if decm.sol.converged:
            log(f'{tag}: DECM converso in {eh} h e {em:.2f} m, '
                f'MRE={decm.sol.mre:.4e} (peak RAM={decm.sol.peak_ram_bytes // 1024**2} MB)')
        else:
            log(f'{tag}: DECM non converso in {eh} h e {em:.2f} m, '
                f'best MRE={decm.sol.mre:.4e} (peak RAM={decm.sol.peak_ram_bytes // 1024**2} MB)')

        with open(ckpt_path, 'wb') as f:
            pickle.dump(decm, f)
        log(f'{tag}: saved to {ckpt_path}')

    except Exception as e:
        log(f'{tag}: ERROR while solving: {e!r}')
        raise


if __name__ == '__main__':
    main()
