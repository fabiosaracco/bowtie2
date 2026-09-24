import os, sys
import platform
import numpy as np
import pickle
import datetime as dt
from collections import defaultdict

from dcms.models import DWCMModel

from auxiliary_functions import el2ks

# Set HOME directory based on the operating system
if platform.system() == 'Darwin':
    HOME = '/Users/fabio/Documents/Lavoro/PythonFiles/bowtie2_py310/bowtie2/'
elif platform.system() == 'Linux':
    HOME = '/home/sarawalk/bowtie2_py39/bowtie2/'
else:
    raise RuntimeError(f"Unsupported OS: {platform.system()}")

sys.path.insert(0, HOME)
DATA_FOLDER = HOME + 'dati_elezioni/'

MAX_TIME_HOURS = 5
MAX_ITER = 5000
TOL = 1e-5
ANDERSON = 7
MONITOR = False


def log(msg):
    print(f'[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}')
    sys.stdout.flush()


def main():
    files = sorted(f for f in os.listdir(DATA_FOLDER) if f.endswith('.csv'))

    for i in range(len(files) // 2):
        dico_file = files[2 * i]
        el_file = files[2 * i + 1]
        dataset_name = dico_file[:-10]
        log(f'***{dataset_name.title()}***')

        dico = np.genfromtxt(
            DATA_FOLDER + dico_file, delimiter=',', skip_header=1, autostrip=True,
            encoding='latin-1',
            dtype=[('user_id', '>U50'), ('dico', '>U2'), ('h_dico', 'U2'), ('i_dico', 'U2')],
        )
        el = np.genfromtxt(
            DATA_FOLDER + el_file, delimiter=',', skip_header=1, autostrip=True,
            encoding='latin-1',
            dtype=[('source_id', '>U50'), ('target_id', '>U20'), ('weight', 'i4')],
        )

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

        el_dico = defaultdict(
            lambda: np.empty(0, dtype=el.dtype),
            {k: np.array(v, dtype=el.dtype) for k, v in _tmp.items()},
        )
        del _tmp

        dicos = sorted(el_dico.keys())

        for dico_class in dicos:
            log(f'Processing DiCo class {dico_class}...')
            aux = el2ks(el_dico[dico_class])

            assert aux[0].sum() == aux[1].sum() == len(el_dico[dico_class])
            assert aux[2].sum() == aux[3].sum()

            log(f'N(nodes)={len(aux[4]):,}, N(edges)={len(el_dico[dico_class]):,}, '
                f'density={len(el_dico[dico_class]) / len(aux[4]) ** 2:.2e}')

            file_name = HOME + f'tests/{dataset_name}_dico{dico_class}_dwcm_final.pkl'

            if os.path.exists(file_name):
                with open(file_name, 'rb') as f:
                    old_dwcm = pickle.load(f)
                if hasattr(old_dwcm, 'sol') and old_dwcm.sol.converged and old_dwcm.sol.mre < TOL:
                    log('Already converged. Skipping...')
                    continue

            log(f'DWCM, pytorch, theta (max: {MAX_TIME_HOURS} hours)')
            dwcm = DWCMModel(aux[2], aux[3])

            try:
                dwcm.solve_tool(
                    tol=TOL, backend='pytorch', ic='strengths',
                    max_time=MAX_TIME_HOURS * 3600, max_iter=MAX_ITER,
                    verbose=True, monitor=MONITOR, anderson_depth=ANDERSON,
                )

                t_ets = dwcm.sol.elapsed_time
                eth, etm = int(t_ets // 3600), (t_ets % 3600) / 60

                if dwcm.sol.converged:
                    log(f'DWCM converged in {eth} h and {etm:.2f} m, MRE={dwcm.sol.mre:.4e} '
                        f'(peak RAM={dwcm.sol.peak_ram_bytes // 1024**2} MB)')
                else:
                    log(f'DWCM did not converge in {eth} h and {etm:.2f} m, MRE={dwcm.sol.mre:.4e} '
                        f'(peak RAM={dwcm.sol.peak_ram_bytes // 1024**2} MB)')

                final_file_name = file_name
                counter = 0
                if os.path.exists(file_name):
                    final_file_name = file_name.replace('.pkl', f'_{counter}.pkl')
                while os.path.exists(final_file_name):
                    counter += 1
                    final_file_name = file_name.replace('.pkl', f'_{counter}.pkl')

                with open(final_file_name, 'wb') as f:
                    pickle.dump(dwcm, f)
                log(f'Saved to {final_file_name}')

            except Exception as e:
                log(f'Error solving DWCM for {dataset_name}_dico{dico_class}: {e!r}')


if __name__ == "__main__":
    main()
    if platform.system() == 'Darwin':
        os.system("afplay ta-da.mp3")
