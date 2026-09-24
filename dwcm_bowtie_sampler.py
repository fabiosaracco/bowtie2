"""Driver: enumerate all (dataset, dico) bowtie2 networks and sample each
one's bowtie p-values in its own fresh subprocess -- see
dwcm_bowtie_sampler_one.py's docstring for why (macOS fork-safety crash
when a single long-lived process creates more than one ProcessPoolExecutor
across successive validate() calls)."""
import os, sys, subprocess
import platform
import datetime as dt
import numpy as np
from collections import defaultdict

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
PVALUE_FOLDER=HOME+'pvalues/'
ONE_SCRIPT=HOME+'dwcm_bowtie_sampler_one.py'


def main():
    files=[f for f in os.listdir(DATA_FOLDER) if f.endswith('.csv')]
    files.sort()

    for i in range(len(files)//2):
        dico_file = files[2*i]
        el_file = files[2*i + 1]
        dataset_name=dico_file[:-10]
        print(f'[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] ***{dataset_name.title()}***')

        dico=np.genfromtxt(DATA_FOLDER+dico_file, delimiter=',',skip_header=1, autostrip=True, encoding='latin-1', dtype=[('user_id', '>U50'), ('dico', '>U2'), ('h_dico', 'U2'), ('i_dico', 'U2')])
        el=np.genfromtxt(DATA_FOLDER+el_file, delimiter=',', skip_header=1,autostrip=True, encoding='latin-1', dtype=[('source_id', '>U50'), ('target_id', '>U20'),('weight', 'i4')])

        dico_dict={}
        for d in dico:
            if d['dico'].isnumeric():
                dico_dict[d['user_id']]=int(d['dico'])

        _tmp = defaultdict(list)
        for edge in el:
            src = edge['source_id'].strip()
            tgt = edge['target_id'].strip()
            d_src = dico_dict.get(src)
            if d_src is not None and d_src == dico_dict.get(tgt):
                _tmp[d_src].append(edge)

        dicos=sorted(_tmp.keys())

        for d in dicos:
            pvalue_block_filename=PVALUE_FOLDER+f'{dataset_name}_dico{d}_dwcm_pvalues_blocks.pkl'
            if os.path.exists(pvalue_block_filename):
                print(f'[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] {dataset_name}_dico{d}: '
                      f'p-values already exist, skipping.')
                sys.stdout.flush()
                continue
            print(f'[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] --- dispatching DiCo {d} to subprocess ---')
            sys.stdout.flush()
            ret = subprocess.run(
                [sys.executable, ONE_SCRIPT, '--dataset', dataset_name, '--dico', str(d)]
            )
            if ret.returncode != 0:
                print(f'[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] {dataset_name}_dico{d}: '
                      f'subprocess exited with code {ret.returncode}, continuing to next network.')
                sys.stdout.flush()

if __name__ == "__main__":
    main()
