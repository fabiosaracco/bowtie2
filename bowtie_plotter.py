# ============================================================
# Preamble (from Intro section of 0_3_bipartite_bowtie.ipynb)
# ============================================================

# --- Global parameters ---
MAX_TIME_HOURS = 1
ALPHA = 0.01
_PVAL_FLOOR = 10**-6
OLD_BIDCM=True

# --- Standard modules ---
import os, pickle, platform, sys
import numpy as np

from collections import defaultdict

import dcms
from dcms.models import DCMModel, DECMModel, qDECMModel, DWCMModel

import matplotlib.pyplot as plt
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['xtick.major.size'] = 10
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['ytick.major.size'] = 10
plt.rcParams['ytick.major.width'] = 2

plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

plt.rcParams['xtick.minor.size'] = 5
plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['ytick.minor.size'] = 5
plt.rcParams['ytick.minor.width'] = 1
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
from matplotlib.ticker import ScalarFormatter

from scipy.stats import spearmanr

from tqdm import tqdm, trange

import datetime as dt

from bowtie import edges2bowtie

# --- Home-made modules ---
if platform.system() == 'Darwin':
    HOME = '/Users/fabio/Documents/Lavoro/PythonFiles/bowtie2_py310/bowtie2/'
elif platform.system() == 'Linux':
    HOME = '/home/sarawalk/bowtie2_py39/bowtie2/'
else:
    raise RuntimeError(f"Unsupported OS: {platform.system()}")

sys.path.insert(0, HOME)

from auxiliary_functions import el2ks

from sam_bowtie import block_and_fluxes as bnf

from plot_bowtie import plot_bowtie_blocks, plot_bowtie_fluxes, _add_colorbar
from plot_bowtie import _fdr as fdr


# --- Data folders ---
DATA_FOLDER = HOME + 'dati_elezioni/'
TEST_FOLDER = HOME + 'tests/'
PVALUE_FOLDER = HOME + 'pvalues/'
GUARINO_FOLDER = HOME + 'guarino_files/'
if OLD_BIDCM:
    BIPARTITE_FOLDER = HOME + 'BiDCM/'
else:
    BIPARTITE_FOLDER = HOME + 'BiDCM2/'
PLOT_FOLDER = HOME + 'plots/'


# ============================================================
# Functions (from ## Functions section of 0_3_bipartite_bowtie.ipynb)
# ============================================================

def guarino2dict_blocks(dataset, dico):
    # get the file
    file_name = f'{dataset}_dico_{dico}_bowtie_sizes.csv'
    # load the data
    cacca = np.genfromtxt(BIPARTITE_FOLDER + file_name, delimiter=',', dtype=int, skip_header=1)
    header = np.genfromtxt(BIPARTITE_FOLDER + file_name, delimiter=',', dtype=str, max_rows=1)
    header = [str(h) for h in header]
    # load the "monopartite" dict
    block_dict_0, flux_dict_0 = ppmf(dataset, dico)

    block_dict_bipartite = {}

    guarino_translator = {'LSCC': 'SCC', 'IN-TENDRILS': 'INTENDRILS', 'OUT-TENDRILS': 'OUTTENDRILS'}

    for i, name in enumerate(header):
        key = guarino_translator.get(name, name)
        block_dict_bipartite[key] = {}
        block_dict_bipartite[key]['obs'] = block_dict_0[key]['obs']
        block_dict_bipartite[key]['sample'] = cacca[:, i]
        _ge = np.sum(cacca[:, i] >= block_dict_bipartite[key]['obs'])
        _le = np.sum(cacca[:, i] <= block_dict_bipartite[key]['obs'])
        block_dict_bipartite[key]['p_value'] = 2 * min(_ge, _le) / len(block_dict_bipartite[key]['sample'])
        _median = np.median(block_dict_bipartite[key]['sample'])
        if _median > block_dict_bipartite[key]['obs']:
            block_dict_bipartite[key]['tail'] = 'left'
        else:
            block_dict_bipartite[key]['tail'] = 'right'
    return block_dict_bipartite, block_dict_0, flux_dict_0


def guarino2dict_fluxes(dataset, dico):
    # get the file
    file_name = f'{dataset}_dico_{dico}_bowtie_flows.csv'
    # load the data
    cacca = np.genfromtxt(BIPARTITE_FOLDER + file_name, delimiter=',', skip_header=1)
    cacca = cacca.astype(int)
    header = np.genfromtxt(BIPARTITE_FOLDER + file_name, delimiter=',', dtype=str, max_rows=1)

    _, flux_dict_0 = ppmf(dataset, dico)

    flux_dict_bipartite = defaultdict(dict)
    guarino_translator = {'LSCC': 'SCC', 'IN-TENDRILS': 'INTENDRILS', 'OUT-TENDRILS': 'OUTTENDRILS'}

    for i, name in enumerate(header):
        source, target = name.split('->')
        # translate the source and target to match the keys in flux_dict_0
        source = guarino_translator.get(source, source)
        target = guarino_translator.get(target, target)
        key = '->'.join([source, target])

        if key not in flux_dict_0.keys():
            continue  # skip if the key is not in flux_dict_0
        flux_dict_bipartite[key]['obs'] = flux_dict_0[key]['obs']
        flux_dict_bipartite[key]['sample'] = cacca[:, i]
        _ge = np.sum(cacca[:, i] >= flux_dict_bipartite[key]['obs'])
        _le = np.sum(cacca[:, i] <= flux_dict_bipartite[key]['obs'])
        flux_dict_bipartite[key]['p_value'] = 2 * min(_ge, _le) / len(flux_dict_bipartite[key]['sample'])
        _median = np.median(flux_dict_bipartite[key]['sample'])
        if _median > flux_dict_bipartite[key]['obs']:
            flux_dict_bipartite[key]['tail'] = 'left'
        else:
            flux_dict_bipartite[key]['tail'] = 'right'

    return flux_dict_bipartite, flux_dict_0


def right_tailer(block_dict):
    right_tailed_dict = {}
    for key, item in block_dict.items():
        right_tailed_dict[key] = item
        if item['tail'] == 'right':
            right_tailed_dict[key]['p_value'] /= 2
        else:
            right_tailed_dict[key]['p_value'] = 1.
    return right_tailed_dict


def ppmf(dataset, dico):
    with open(PVALUE_FOLDER + f'{dataset}_dico{dico}_pvalues_blocks_1.pkl', 'rb') as f:
        block_dict_1 = pickle.load(f)

    with open(PVALUE_FOLDER + f'{dataset}_dico{dico}_pvalues_fluxes_1.pkl', 'rb') as f:
        flux_dict_1 = pickle.load(f)

    for key, value in block_dict_1.items():
        block_dict_1[key]['mean_sim'] = np.mean(value['count_sample'])
        block_dict_1[key]['std_sim'] = np.std(value['count_sample'])
        if np.median(value['count_sample']) > block_dict_1[key]['obs']:
            block_dict_1[key]['tail'] = 'left'
        else:
            block_dict_1[key]['tail'] = 'right'

    for key, value in flux_dict_1.items():
        flux_dict_1[key]['mean_sim'] = np.mean(value['count_sample'])
        flux_dict_1[key]['std_sim'] = np.std(value['count_sample'])
        if np.median(value['count_sample']) > flux_dict_1[key]['obs']:
            flux_dict_1[key]['tail'] = 'left'
        else:
            flux_dict_1[key]['tail'] = 'right'

    return block_dict_1, flux_dict_1


def guarino2dict_blocks_DCM(dataset, dico):
    # get the file
    file_name = f'{dataset}_dico_{dico}_bowtie_sizes.csv'
    # load the data
    cacca = np.genfromtxt(GUARINO_FOLDER + file_name, delimiter=',', dtype=int, skip_header=1)
    header = np.genfromtxt(GUARINO_FOLDER + file_name, delimiter=',', dtype=str, max_rows=1)
    header = [str(h) for h in header]
    # load the "monopartite" dict
    with open(PVALUE_FOLDER + f'{dataset}_dico{dico}_pvalues_blocks_0.pkl', 'rb') as f:
        block_dict_0 = pickle.load(f)

    with open(PVALUE_FOLDER + f'{dataset}_dico{dico}_pvalues_fluxes_0.pkl', 'rb') as f:
        flux_dict_0 = pickle.load(f)

    block_dict_bipartite = {}

    for i, name in enumerate(header):
        if name == 'LSCC':
            key = 'SCC'
        else:
            key = name.replace('-', '')
        block_dict_bipartite[key] = {}
        block_dict_bipartite[key]['obs'] = block_dict_0[key]['obs']
        block_dict_bipartite[key]['sample'] = cacca[:, i]
        _ge = np.sum(cacca[:, i] >= block_dict_bipartite[key]['obs'])
        _le = np.sum(cacca[:, i] <= block_dict_bipartite[key]['obs'])
        block_dict_bipartite[key]['p_value'] = 2 * min(_ge, _le) / len(block_dict_bipartite[key]['sample'])
        _median = np.median(block_dict_bipartite[key]['sample'])
        if _median > block_dict_bipartite[key]['obs']:
            block_dict_bipartite[key]['tail'] = 'left'
        else:
            block_dict_bipartite[key]['tail'] = 'right'
    return block_dict_bipartite, block_dict_0, flux_dict_0

def fluxes_plotter(fluxes, filename):
    '''
    The function plots the right tail validation of the fluxes of the bowtie
    '''

    # handle minimum p-value for color scaling
    vmin_fluxes_0 =min([fluxes[0][key]['p_value'] for key in fluxes[0].keys()])
    vmin_fluxes_1 =min([fluxes[1][key]['p_value'] for key in fluxes[1].keys()])
    vmin_fluxes=min(vmin_fluxes_0, vmin_fluxes_1)

    fdr_fluxes_0 =fdr([fluxes[0][key]['p_value'] for key in fluxes[0].keys()], 0.01)
    fdr_fluxes_1 =fdr([fluxes[1][key]['p_value'] for key in fluxes[1].keys()], 0.01)
    fdr_fluxes=[fdr_fluxes_0, fdr_fluxes_1]
    m_fdr_th=min(fdr_fluxes)

    vmin = 10 ** np.floor(np.log10(m_fdr_th))
    vmin = min(vmin_fluxes, vmin)
    vmin = max(_PVAL_FLOOR, vmin)

    # define the figure and axes
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
    plot_bowtie_fluxes(fluxes[1], 0.01, ax=axs[0], show_colorbar=False, vmin=vmin)
    plot_bowtie_fluxes(fluxes[0], 0.01, ax=axs[1], show_colorbar=False, vmin=vmin)


    bnorm = mcolors.LogNorm(vmin=vmin, vmax=1.0)
    bcmap = plt.get_cmap('cool')
    cb=_add_colorbar(fig, axs, bcmap, bnorm, 'p-value',fdr_th=None, pad=0.02, orientation='vertical')
    colors=['darkmagenta', 'darkorchid']
    models=['qDECM', 'BiDCM']


    x_text = -0.75
    pairs = sorted(zip(fdr_fluxes, models, colors), key=lambda t: t[0], reverse=True)
    epsilon = 0.5 * (fdr_fluxes[1] - fdr_fluxes[0])
    for (fdr_th, model, color), va in zip(pairs, ['bottom', 'top']):
        cb.ax.axhline(y=fdr_th, color=color, linestyle='-', linewidth=1.5)
        if va == 'bottom':
            y_text = fdr_th - epsilon
        else:        
            y_text = fdr_th + epsilon
        cb.ax.plot([x_text, 0], [fdr_th, fdr_th], color=color, linewidth=1.5,
            clip_on=False, transform=cb.ax.get_yaxis_transform())
        cb.ax.text(x_text, y_text, f'FDR th ({model})', color=color,
            ha='center', va=va, fontsize=10, rotation=90,
            clip_on=False, transform=cb.ax.get_yaxis_transform())
    for ax, model in zip(axs, models):
        ax.set_title(model, fontsize=18)
    plt.savefig(filename, dpi=300, bbox_inches='tight')

def get_dico_dict():
    filepath = GUARINO_FOLDER + 'all_dico_labels.txt'
    with open(filepath, 'r') as f:
        lines = [line.strip().split() for line in f]
        # such a command creates a list
        # in which each element is a list of tokens of a line in the file
        # remarkably, there is an empty element
        # before each dataset name
        

    result = {}
    current_key = None

    for tokens in lines:
        if not tokens:
            current_key = None
        elif current_key is None:
            # prima riga non vuota del gruppo: nome del dataset
            current_key = tokens[0]
            result[current_key] = {}
        else:
            # riga tipo ['5:', 'journalists', '&', 'Media']
            idx = int(tokens[0].rstrip(':'))
            label = ' '.join(tokens[1:])
            result[current_key][idx] = label

    return result

def main():

    datasets=['ita_elections', 'crisi', 'quirinale']
    dico_dict = get_dico_dict()

    for dataset in tqdm(datasets):
        dicos = sorted(dico_dict[dataset].keys())
        for dico in tqdm(dicos, leave=False):
            fluxes=guarino2dict_fluxes(dataset, dico)
            rt_fluxes_0 = right_tailer(fluxes[0])
            rt_fluxes_1 = right_tailer(fluxes[1])
            if OLD_BIDCM:
                filename = PLOT_FOLDER+f'{dataset}_{dico}_fluxes_old.png'
            else:
                filename = PLOT_FOLDER+f'{dataset}_{dico}_fluxes.png'
            fluxes_plotter([rt_fluxes_0, rt_fluxes_1], filename)


if __name__ == "__main__":
    main()
