"""Warm-started DECM resume, for the 3 batch2608 networks that only need
more time (quirinale_dico2, crisi_dico1, ita_elections_dico0): all three
showed a monotonically-decreasing best-MRE with only isolated (self-
recovering) blowups when the 2026-08-28 batch hit its 24h cap, not a
stuck/plateaued pattern.

Loads the existing tests/batch2608_{tag}_decm.pkl checkpoint, warm-starts
solve_tool from its sol.best_theta (single run, multi_start ignored per
solve_tool's docs when ic is an array), and gives it another 24h. Overwrites
the same checkpoint on completion (safe: the old checkpoint is never
touched until the new solve_tool call returns).

Same recipe as the original batch (z_clamp=1e-6, noise_base=2e-3, etc.) --
these three never needed the hub-degeneracy machinery to kick in beyond
what already ran, so there's no reason to change parameters, just extend
the clock.
"""
import argparse
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
TEST_FOLDER = HOME + 'tests/'

MAX_TIME_HOURS = 24
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--dico', required=True, type=int)
    args = ap.parse_args()

    dataset, dico_class = args.dataset, args.dico
    tag = f'{dataset}_dico{dico_class}'
    ckpt_path = TEST_FOLDER + f'batch2608_{tag}_decm.pkl'

    with open(ckpt_path, 'rb') as f:
        decm = pickle.load(f)

    if decm.sol.converged and decm.sol.mre < TOL:
        log(f'{tag}: already converged, nothing to resume.')
        return

    warm_theta = decm.sol.best_theta
    prev_mre = decm.sol.mre
    log(f'*** resuming {tag} from best_theta (prev best MRE={prev_mre:.4e}) ***')

    try:
        decm.solve_tool(
            ic=warm_theta,
            tol=TOL,
            max_iter=MAX_ITER,
            max_time=MAX_TIME_HOURS * 3600,
            anderson_depth=ANDERSON,
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
                f'best MRE={decm.sol.mre:.4e} (was {prev_mre:.4e} before resume) '
                f'(peak RAM={decm.sol.peak_ram_bytes // 1024**2} MB)')

        with open(ckpt_path, 'wb') as f:
            pickle.dump(decm, f)
        log(f'{tag}: saved to {ckpt_path}')

    except Exception as e:
        log(f'{tag}: ERROR while resuming: {e!r}')
        raise


if __name__ == '__main__':
    main()
