"""z_clamp diagnostic sweep on ita_elections_dico1, driven by CLI arg.

Context: in the 2026-08-28 batch this network hit severe blowups in the
very first ~70 iterations (res_norm up to ~430x the guard threshold,
res_norm=1.0 baseline) with z_clamp=1e-6, then crawled to a plateau at
MRE~0.136 after 1534 iterations / 24h -- nowhere near tol=1e-5.

z_clamp floors the raw z=eta_out+eta_in sum in two coupled roles: (1) it
damps the G-curvature blowup as z->0 for near-degenerate hub pairs (a
*higher* z_clamp should suppress early blowups more), and (2) it hard-
floors the per-step trust-region limiter on how far eta can approach z=0
in one iteration (a *higher* z_clamp can wall off pairs whose true z*
sits below the new floor, creating an artificial plateau -- this is
documented in dcms' own solve_fixed_point_decm docstring as a real,
previously-observed failure mode). Since both the initial blowup AND the
later plateau are visible in the same run, it's not obvious a priori
which direction (raise or lower relative to the 1e-6 used) helps more --
hence sweeping both ways instead of guessing.

Deliberately fresh start (ic='degrees', multi_start=True), NOT a warm
start from the batch2608 checkpoint: we want to see whether a different
z_clamp changes the *early* blowup behaviour, which a warm start from
past the blowup region would hide. Deliberately capped at a few hours
(not 24h): this is a diagnostic to compare trajectories, not a run meant
to reach convergence.
"""
import argparse
import os
import sys
import platform
import pickle
import datetime as dt
from collections import defaultdict

import numpy as np

if platform.system() == 'Darwin':
    HOME = '/Users/fabio/Documents/Lavoro/PythonFiles/bowtie2_py310/bowtie2/'
elif platform.system() == 'Linux':
    HOME = '/home/sarawalk/bowtie2_py39/bowtie2/'
else:
    raise RuntimeError(f"Unsupported OS: {platform.system()}")

sys.path.insert(0, HOME)
from dcms.models import DECMModel
from auxiliary_functions import el2ks
from decm_dico_calculator_batch2608 import load_dico_edges

TEST_FOLDER = HOME + 'tests/'

TOL = 1e-5
NOISE_BASE = 2e-3
HUB_TH = 5.0
GAMMA = 0.0
ANDERSON = 10
MAX_ITER = 300000

DATASET = 'ita_elections'
DICO = 1


def log(msg):
    print(f'[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}')
    sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zclamp', required=True, type=float)
    ap.add_argument('--hours', required=True, type=float)
    args = ap.parse_args()

    z_clamp = args.zclamp
    tag = f'{DATASET}_dico{DICO}_zsweep_{z_clamp:.0e}'
    ckpt_path = TEST_FOLDER + f'{tag}_decm.pkl'

    log(f'*** {tag} (z_clamp={z_clamp:.1e}, max {args.hours}h) ***')
    el_dico = load_dico_edges(DATASET, DICO)
    k_out, k_in, s_out, s_in, nodes = el2ks(el_dico)

    decm = DECMModel(k_out, k_in, s_out, s_in)

    try:
        decm.solve_tool(
            ic='degrees',
            tol=TOL,
            max_iter=MAX_ITER,
            max_time=args.hours * 3600,
            anderson_depth=ANDERSON,
            multi_start=True,
            backend='auto',
            num_threads=1,
            verbose=True,
            monitor=False,
            hub_sk_threshold=HUB_TH,
            backtracking_gamma=GAMMA,
            z_clamp=z_clamp,
            reduce_degeneracy=True,
            noise_base=NOISE_BASE,
        )

        t = decm.sol.elapsed_time
        eh, em = int(t // 3600), (t % 3600) / 60

        if decm.sol.converged:
            log(f'{tag}: DECM converso in {eh} h e {em:.2f} m, MRE={decm.sol.mre:.4e}')
        else:
            log(f'{tag}: DECM non converso in {eh} h e {em:.2f} m, best MRE={decm.sol.mre:.4e}')

        with open(ckpt_path, 'wb') as f:
            pickle.dump(decm, f)
        log(f'{tag}: saved to {ckpt_path}')

    except Exception as e:
        log(f'{tag}: ERROR while solving: {e!r}')
        raise


if __name__ == '__main__':
    main()
