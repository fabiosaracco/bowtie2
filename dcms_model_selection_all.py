import os, sys, pickle
import platform
import datetime as dt
import time
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

def main():


def final_results_gatherer(folder):
    """
    Gather the results of the runs and save them in a single file.
    """
    # Load the results of the runs
    results = [pickle.load(open(f'{folder}/{f}', 'rb')) for f in os.listdir(folder) if f.endswith('.pkl')]

    for i_r, r in enumerate(results):
        
        
    # Save the results in a single file
    with open(f'{PVALUE_FOLDER}final_results.pkl', 'wb') as f:
        pickle.dump(results, f)




if __name__ == "__main__":
    if platform.system() == 'Darwin':
        print(f'[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] Air!')
    elif platform.system() == 'Linux':
        print(f'[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] Stella!')

    main()
    if platform.system() == 'Darwin':
        os.system("afplay ta-da.mp3")
