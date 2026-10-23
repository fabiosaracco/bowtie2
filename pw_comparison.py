import os, pickle, platform, sys
import numpy as np
import copy
import torch

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

from tqdm.autonotebook import tqdm, trange
import datetime as dt

import argparse


if platform.system() == 'Darwin':
    print('Air!')
    HOME = '/Users/fabio/Documents/Lavoro/PythonFiles/bowtie2_py310/bowtie2/'
elif platform.system() == 'Linux':
    print('Stella!')
    HOME = '/home/sarawalk/bowtie2_py39/bowtie2/'
else:
    raise RuntimeError(f"Unsupported OS: {platform.system()}")

sys.path.insert(0, HOME)

# Folders
DATA_FOLDER=HOME+'dati_elezioni/'
TEST_FOLDER=HOME+'tests/'
PVALUE_FOLDER=HOME+'pvalues/'
GUARINO_FOLDER=HOME+'guarino_files/'
BIPARTITE_FOLDER=HOME+'BiDCM/'
PLOT_FOLDER=HOME+'plots/'


def get_converged_solutions():
    solution_files=[f for f in os.listdir(TEST_FOLDER) if f.endswith('.pkl')]
    solution_files.sort()
    final_dict={}
    for solution in solution_files:
        # get the model type from the solution filename
        if 'qdecm' in solution.lower():
            model_type='qDECM'
        elif 'decm' in solution.lower():
            model_type='DECM'
        elif 'dwcm' in solution.lower():
            model_type='DWCM'
        else:
            continue

        # get the dataset from the solution filename
        if 'elections' in solution.lower():
            name='e'
        elif 'crisi' in solution.lower():
            name='c'
        elif 'quirinale' in solution.lower():
            name='p'
        else:
            continue
        
        try:
            i_dico=solution.index('dico')
            dico=solution[i_dico+4]
        except ValueError:
            continue

        dataset=f"{name}{dico}"
        if os.path.exists(TEST_FOLDER+solution):
            try:
                with open(TEST_FOLDER+solution, 'rb') as f:
                    _temp = pickle.load(f)
            except Exception:
                os.remove(TEST_FOLDER+solution)
                continue
            try:
                if _temp.sol.mre < 1e-5:  # Check if the model converged
                    if dataset not in final_dict:
                        final_dict[dataset]={}
                    if model_type not in final_dict[dataset]:
                        final_dict[dataset][model_type]=_temp
            except AttributeError:
                pass
    return final_dict

def status(dataset, i, n, msg, width=50):
    print(f"[{dataset}, {i:2d}/{n:2d}] {msg:<{width}}", end='\r')

def plot_p_comparison(final_dict, rng, sample_size, colors, labels_decm, labels_qdecm, labels_dwcm, verbose, i, p_or_w):
    keys=list(final_dict.keys())
    dataset=keys[i]
    
    
    sol_dict=final_dict[dataset]

    # Check if all three models are present
    if not all(model in sol_dict for model in ['DECM', 'qDECM', 'DWCM']):
        if verbose:
            msg="Dataset does not have all three models. Skipping."
            status(dataset, i, len(final_dict), msg)
            return 

    # Extract solutions
    decm_sol=sol_dict['DECM']
    qdecm_sol=sol_dict['qDECM']
    dwcm_sol=sol_dict['DWCM']
    N=decm_sol.N

    if N**2>2*10**9:
        msg=f"N={N:.2e} is too large, skipping dataset..."
        status(dataset, i, len(final_dict), msg)
        return 1
    if verbose:
        msg=f"N={N:.2e} Processing dataset..."
        status(dataset, i, len(final_dict), msg)
    # Now you can proceed with your analysis using decm_sol, qdecm_sol, and dwcm_sol
    theta_decm=decm_sol.sol.best_theta
    theta_qdecm=qdecm_sol.sol.best_theta
    theta_dwcm=dwcm_sol.sol.best_theta

    # select some of the entries
    if verbose:
        msg=f"N={N:.2e} Selecting entries..."
        status(dataset, i+1, len(final_dict), msg)   

    selected_points=rng.choice(N**2, size=sample_size, replace=False)
    
    if p_or_w=='p':
        # probability matrices
        if verbose:
            msg="Processing DECM..."
            status(dataset, i+1, len(final_dict), msg)    
        p_decm=decm_sol.pij_matrix(theta_decm).flatten()
        p_decm=p_decm[selected_points]

        if verbose:
            msg="Processing qDECM..."
            status(dataset, i+1, len(final_dict), msg)    
        p_qdecm=qdecm_sol.pij_matrix(theta_qdecm[:2*N]).flatten()
        p_qdecm=p_qdecm[selected_points]

        if verbose:
            msg="Processing DWCM..."
            status(dataset, i+1, len(final_dict), msg)        
        p_dwcm=dwcm_sol.pij_matrix(theta_dwcm).flatten()
        p_dwcm=p_dwcm[selected_points]

        max_p=np.max([p_decm.max(), p_qdecm.max(), p_dwcm.max()])

        if verbose:
            msg="Points selected..."
            status(dataset, i+1, len(final_dict), msg)

    elif p_or_w=='w':
            # probability matrices
            if verbose:
                msg="Processing DECM..."
                status(dataset, i+1, len(final_dict), msg)    
            p_decm=decm_sol.wij_matrix(theta_decm).flatten()
            p_decm=p_decm[selected_points]
    
            if verbose:
                msg="Processing qDECM..."
                status(dataset, i+1, len(final_dict), msg)    
            p_qdecm=qdecm_sol.wij_matrix_conditioned(theta_qdecm[:2*N], theta_qdecm[2*N:]).flatten()
            p_qdecm=p_qdecm[selected_points]
    
            if verbose:
                msg="Processing DWCM..."
                status(dataset, i+1, len(final_dict), msg)        
            p_dwcm=dwcm_sol.wij_matrix(theta_dwcm).flatten()
            p_dwcm=p_dwcm[selected_points]
    
            max_p=np.max([p_decm.max(), p_qdecm.max(), p_dwcm.max()])
    
            if verbose:
                msg="Points selected..."
                status(dataset, i+1, len(final_dict), msg)
    else:
        raise ValueError(f"Invalid value for p_or_w: {p_or_w}. Must be 'p' or 'w'.")
    # plot
    fig, axs= plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True, sharex=True, sharey=True)

    axs[0].scatter(p_decm, p_dwcm, color=colors[0])
    axs[0].plot([-1,2*max_p], [-1,2*max_p], '--', color='gray')
    axs[0].set_xlim([-0.05, max_p*1.05])
    axs[0].set_ylim([-0.05, max_p*1.05])
    #axs[0].set_xscale('log')
    #axs[0].set_yscale('log')    

    axs[1].scatter(p_decm, p_qdecm, color=colors[1])
    axs[1].plot([-1,2*max_p], [-1,2*max_p], '--', color='gray')
    axs[1].set_xlim([-0.05, max_p*1.05])
    axs[1].set_ylim([-0.05, max_p*1.05])
    #axs[1].set_xscale('log')
    #axs[1].set_yscale('log')    

    if p_or_w=='p':
        label_index=0 
    else:
        label_index=1
    for j in range(2):
        axs[j].set_xlabel(labels_decm[label_index], fontsize=16)
    axs[0].set_ylabel(labels_dwcm[label_index], fontsize=16)
    axs[1].set_ylabel(labels_qdecm[label_index], fontsize=16)
    plt.savefig(PLOT_FOLDER+f"{p_or_w}_comparison_{dataset}.png")
    if verbose:
        msg="Plot saved!"
        status(dataset, i+1, len(final_dict), msg)    
    return 0




def main():
    # Load the converged solutions
    final_dict = get_converged_solutions()

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Plot comparison of p and w values')
    parser.add_argument('p_or_w', choices=['p', 'w'], help='Whether to plot p or w values')
    args = parser.parse_args()

    # Set random seed for reproducibility
    rng = np.random.default_rng()

    # Define parameters for plotting
    sample_size = 10**4  # Number of entries to sample for plotting
    if args.p_or_w == 'p':
        colors=['orchid','darkorchid']
    else:
        colors=['magenta','darkmagenta']
    labels_decm=[r'$p_{DECM}$', r'$\langle w\rangle_{DECM}$']
    labels_qdecm=[r'$p_{qDECM}$', r'$\langle w\rangle_{qDECM}$']
    labels_dwcm=[r'$p_{DWCM}$', r'$\langle w\rangle_{DWCM}$']
    
    # Loop through each dataset and plot comparisons
    for i in range(len(final_dict)):
        if os.path.exists(PLOT_FOLDER+f"{args.p_or_w}_comparison_{list(final_dict.keys())[i]}.png"):
            continue
        plot_p_comparison(final_dict, rng, sample_size, colors, labels_decm, labels_qdecm, labels_dwcm, verbose=True, i=i, p_or_w=args.p_or_w)  

if __name__ == "__main__":
    main()
    