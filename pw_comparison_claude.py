import os, pickle, platform, sys
import numpy as np
import torch

import dcms
from dcms.models import DCMModel, DECMModel, qDECMModel, DWCMModel

import matplotlib
matplotlib.use('Agg')
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

# Clamps used by the *_matrix methods of the model classes (kept identical here)
_Z_G_CLAMP_DECM=1e-6    # dcms/models/decm.py, _Z_G_CLAMP
_Z_CLAMP_QDECM=1e-8     # dcms/models/qdecm.py, wij_matrix_conditioned
_Z_CLAMP_DWCM=1e-15     # dcms/models/dwcm.py, wij_matrix


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


# ----------------------------------------------------------------------
# Pairwise p_ij and <w_ij>: same formulas (and zero-degree/zero-strength
# masks) as the pij_matrix/wij_matrix methods, evaluated only on the
# selected pairs (i[k], j[k]) instead of on the whole N x N matrix.
# ----------------------------------------------------------------------

def _to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.to(device='cpu', dtype=torch.float64)
    return torch.tensor(x, dtype=torch.float64, device='cpu')

def pij_pairs_dcm(dcm_model, theta_topo, i, j):
    """DCMModel.pij_matrix(theta_topo)[i, j]"""
    theta_topo=_to_tensor(theta_topo)
    N=dcm_model.N
    theta_out=theta_topo[:N]
    theta_in=theta_topo[N:]
    P=torch.sigmoid(-theta_out[i]-theta_in[j])
    P[i==j]=0.0
    P[dcm_model.zero_out[i]]=0.0
    P[dcm_model.zero_in[j]]=0.0
    return P

def pij_pairs_decm(decm_model, theta, i, j):
    """DECMModel.pij_matrix(theta)[i, j]"""
    theta=_to_tensor(theta)
    N=decm_model.N
    theta_out=theta[:N]
    theta_in=theta[N:2*N]
    eta_out=theta[2*N:3*N]
    eta_in=theta[3*N:]

    eta_safe=(eta_out[i]+eta_in[j]).clamp(min=_Z_G_CLAMP_DECM)
    log_q=-torch.log(torch.expm1(eta_safe))
    P=torch.sigmoid(-theta_out[i]-theta_in[j]+log_q)
    P[i==j]=0.0
    P[decm_model.zero_k_out[i]]=0.0
    P[decm_model.zero_k_in[j]]=0.0
    return P

def wij_pairs_decm(decm_model, theta, i, j):
    """DECMModel.wij_matrix(theta)[i, j]"""
    theta=_to_tensor(theta)
    N=decm_model.N
    eta_out=theta[2*N:3*N]
    eta_in=theta[3*N:]

    eta_safe=(eta_out[i]+eta_in[j]).clamp(min=_Z_G_CLAMP_DECM)
    G=-1.0/torch.expm1(-eta_safe)
    G[i==j]=0.0
    G[decm_model.zero_s_out[i]]=0.0
    G[decm_model.zero_s_in[j]]=0.0
    return pij_pairs_decm(decm_model, theta, i, j)*G

def pij_pairs_qdecm(qdecm_model, theta_topo, i, j):
    """qDECMModel.pij_matrix(theta_topo)[i, j]"""
    return pij_pairs_dcm(qdecm_model._dcm, theta_topo, i, j)

def wij_pairs_qdecm(qdecm_model, theta_topo, theta_weight, i, j):
    """qDECMModel.wij_matrix_conditioned(theta_topo, theta_weight)[i, j]"""
    theta_weight=_to_tensor(theta_weight)
    N=qdecm_model.N
    theta_b_out=theta_weight[:N]
    theta_b_in=theta_weight[N:]

    z_safe=(theta_b_out[i]+theta_b_in[j]).clamp(min=_Z_CLAMP_QDECM)
    G=-1.0/torch.expm1(-z_safe)
    G[i==j]=0.0
    G[qdecm_model.zero_s_out[i]]=0.0
    G[qdecm_model.zero_s_in[j]]=0.0
    return pij_pairs_qdecm(qdecm_model, theta_topo, i, j)*G

def pij_pairs_dwcm(dwcm_model, theta, i, j):
    """DWCMModel.pij_matrix(theta)[i, j]"""
    theta=_to_tensor(theta)
    N=dwcm_model.N
    theta_out=theta[:N]
    theta_in=theta[N:]
    P=torch.exp(-(theta_out[i]+theta_in[j]))
    P[i==j]=0.0
    P[dwcm_model.zero_out[i]]=0.0
    P[dwcm_model.zero_in[j]]=0.0
    return P

def wij_pairs_dwcm(dwcm_model, theta, i, j):
    """DWCMModel.wij_matrix(theta)[i, j]"""
    theta=_to_tensor(theta)
    N=dwcm_model.N
    theta_out=theta[:N]
    theta_in=theta[N:]
    z_safe=(theta_out[i]+theta_in[j]).clamp(min=_Z_CLAMP_DWCM)
    W=1.0/torch.expm1(z_safe)
    W[i==j]=0.0
    W[dwcm_model.zero_out[i]]=0.0
    W[dwcm_model.zero_in[j]]=0.0
    return W


def plot_p_comparison(final_dict, rng, sample_size, colors, labels_decm, labels_qdecm, labels_dwcm, verbose, i, p_or_w):
    keys=list(final_dict.keys())
    dataset=keys[i]

    sol_dict=final_dict[dataset]

    # Check if all three models are present
    if not all(model in sol_dict for model in ['DECM', 'qDECM', 'DWCM']):
        if verbose:
            msg="Dataset does not have all three models. Skipping."
            status(dataset, i+1, len(final_dict), msg)
        return

    # Extract solutions
    decm_sol=sol_dict['DECM']
    qdecm_sol=sol_dict['qDECM']
    dwcm_sol=sol_dict['DWCM']
    N=decm_sol.N

    if verbose:
        msg=f"N={N:.2e} Processing dataset..."
        status(dataset, i+1, len(final_dict), msg)
    theta_decm=decm_sol.sol.best_theta
    theta_qdecm=qdecm_sol.sol.best_theta
    theta_dwcm=dwcm_sol.sol.best_theta

    # select the pairs first: flat index k in [0, N**2) -> (row, col)=(k//N, k%N)
    if verbose:
        msg=f"N={N:.2e} Selecting entries..."
        status(dataset, i+1, len(final_dict), msg)

    selected_points=rng.choice(N**2, size=sample_size, replace=False)
    rows=torch.as_tensor(selected_points//N, dtype=torch.long)
    cols=torch.as_tensor(selected_points%N, dtype=torch.long)

    # then evaluate the models only on those pairs
    if p_or_w=='p':
        if verbose:
            msg="Processing DECM..."
            status(dataset, i+1, len(final_dict), msg)
        p_decm=pij_pairs_decm(decm_sol, theta_decm, rows, cols).numpy()

        if verbose:
            msg="Processing qDECM..."
            status(dataset, i+1, len(final_dict), msg)
        p_qdecm=pij_pairs_qdecm(qdecm_sol, theta_qdecm[:2*N], rows, cols).numpy()

        if verbose:
            msg="Processing DWCM..."
            status(dataset, i+1, len(final_dict), msg)
        p_dwcm=pij_pairs_dwcm(dwcm_sol, theta_dwcm, rows, cols).numpy()

    elif p_or_w=='w':
        if verbose:
            msg="Processing DECM..."
            status(dataset, i+1, len(final_dict), msg)
        p_decm=wij_pairs_decm(decm_sol, theta_decm, rows, cols).numpy()

        if verbose:
            msg="Processing qDECM..."
            status(dataset, i+1, len(final_dict), msg)
        p_qdecm=wij_pairs_qdecm(qdecm_sol, theta_qdecm[:2*N], theta_qdecm[2*N:], rows, cols).numpy()

        if verbose:
            msg="Processing DWCM..."
            status(dataset, i+1, len(final_dict), msg)
        p_dwcm=wij_pairs_dwcm(dwcm_sol, theta_dwcm, rows, cols).numpy()
    else:
        raise ValueError(f"Invalid value for p_or_w: {p_or_w}. Must be 'p' or 'w'.")

    max_p=np.max([p_decm.max(), p_qdecm.max(), p_dwcm.max()])
    # margins proportional to max_p, so that small-w networks are not squashed at the top

    if verbose:
        msg="Points selected..."
        status(dataset, i+1, len(final_dict), msg)

    # plot
    fig, axs= plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True, sharex=True, sharey=True)

    axs[0].scatter(p_decm, p_dwcm, color=colors[0])
    axs[0].plot([-1,2*max_p], [-1,2*max_p], '--', color='gray')
    axs[0].set_xlim([-0.05*max_p, max_p*1.05])
    axs[0].set_ylim([-0.05*max_p, max_p*1.05])

    axs[1].scatter(p_decm, p_qdecm, color=colors[1])
    axs[1].plot([-1,2*max_p], [-1,2*max_p], '--', color='gray')
    axs[1].set_xlim([-0.05*max_p, max_p*1.05])
    axs[1].set_ylim([-0.05*max_p, max_p*1.05])

    if p_or_w=='p':
        label_index=0
    else:
        label_index=1
    for j in range(2):
        axs[j].set_xlabel(labels_decm[label_index], fontsize=16)
    axs[0].set_ylabel(labels_dwcm[label_index], fontsize=16)
    axs[1].set_ylabel(labels_qdecm[label_index], fontsize=16)
    plt.savefig(PLOT_FOLDER+f"{p_or_w}_comparison_claude_{dataset}.png")
    plt.close(fig)
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
        if os.path.exists(PLOT_FOLDER+f"{args.p_or_w}_comparison_claude_{list(final_dict.keys())[i]}.png"):
            continue
        plot_p_comparison(final_dict, rng, sample_size, colors, labels_decm, labels_qdecm, labels_dwcm, verbose=True, i=i, p_or_w=args.p_or_w)
    print()

if __name__ == "__main__":
    main()
