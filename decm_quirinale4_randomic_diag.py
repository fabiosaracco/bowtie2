"""One-off diagnostic: does quirinale_dico4 land on the same stuck value
(best MRE=4.338e-01) with a genuinely different starting point?

In the 2026-08-28 batch (ic='degrees'), 3 perturbed-restarts around the
best point (noise_scale 2e-3/4e-3/8e-3) all landed on the *exact same*
best=4.338e-01 -- noise-around-the-incumbent isn't escaping whatever
basin this is. This tests ic='random' instead (still with
reduce_degeneracy=True -- the reduction only collapses nodes sharing an
identical (k_out,k_in,s_out,s_in) 4-tuple into one group before solving
the *same* equations, it does not change the fixed point, so it should
not be the cause of a stuck value by itself).

multi_start=False on purpose: we want a single clean read on 'random' as
the primary IC, not a fallback chain that could obscure the result.
Capped at 8h (diagnostic: in the original run the first stall+restart
happened by iter ~1030 and the second by ~3288, so 8h at a similar
~16s/iter should cover at least 2 restart cycles without spending a full
24h on what is meant to be a quick check).
"""
import os
import sys
import platform
import pickle
import datetime as dt

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
Z_CLAMP = 1e-6
NOISE_BASE = 2e-3
HUB_TH = 5.0
GAMMA = 0.0
ANDERSON = 10
MAX_ITER = 300000
MAX_TIME_HOURS = 8

DATASET = 'quirinale'
DICO = 4
TAG = f'{DATASET}_dico{DICO}_randomic_diag'


def log(msg):
    print(f'[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}')
    sys.stdout.flush()


def main():
    ckpt_path = TEST_FOLDER + f'{TAG}_decm.pkl'

    log(f'*** {TAG}: ic=random, reduce_degeneracy=True, max {MAX_TIME_HOURS}h ***')
    el_dico = load_dico_edges(DATASET, DICO)
    k_out, k_in, s_out, s_in, nodes = el2ks(el_dico)

    decm = DECMModel(k_out, k_in, s_out, s_in)

    try:
        decm.solve_tool(
            ic='random',
            tol=TOL,
            max_iter=MAX_ITER,
            max_time=MAX_TIME_HOURS * 3600,
            anderson_depth=ANDERSON,
            multi_start=False,
            backend='auto',
            num_threads=1,
            verbose=True,
            monitor=False,
            hub_sk_threshold=HUB_TH,
            backtracking_gamma=GAMMA,
            z_clamp=Z_CLAMP,
            reduce_degeneracy=True,
            noise_base=NOISE_BASE,
            seed=12345,
        )

        t = decm.sol.elapsed_time
        eh, em = int(t // 3600), (t % 3600) / 60

        if decm.sol.converged:
            log(f'{TAG}: DECM converso in {eh} h e {em:.2f} m, MRE={decm.sol.mre:.4e}')
        else:
            log(f'{TAG}: DECM non converso in {eh} h e {em:.2f} m, best MRE={decm.sol.mre:.4e} '
                f'(batch2608 ic=degrees landed on best MRE=4.338e-01, for comparison)')

        with open(ckpt_path, 'wb') as f:
            pickle.dump(decm, f)
        log(f'{TAG}: saved to {ckpt_path}')

    except Exception as e:
        log(f'{TAG}: ERROR while solving: {e!r}')
        raise


if __name__ == '__main__':
    main()
