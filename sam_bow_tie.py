import os, sys, pickle
import datetime as dt
import numpy as np
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

from bowtie import edges2bowtie


def _worker_chunk(args):
    """Run n_chunk independent sampling iterations and return raw block/flux lists.
    Each worker reseeds numpy to avoid correlated samples across processes.
    """
    sol, n_chunk, seed = args
    np.random.seed(seed)
    blocks_list = []
    fluxes_list = []
    for _ in range(n_chunk):
        sampled_wel = sol.sample()
        sim_blocks, sim_fluxes = block_and_fluxes(sampled_wel)
        blocks_list.append(dict(sim_blocks))
        fluxes_list.append(dict(sim_fluxes))
    return blocks_list, fluxes_list


def validate(weighted_el, sol, n_runs, n_workers=None):
    emp_bowtie_blocks, emp_bowtie_fluxes=block_and_fluxes(weighted_el)
    block_dict=defaultdict(dict)
    flux_dict=defaultdict(dict)
    for block, dim_block in emp_bowtie_blocks.items():
        block_dict[block]['obs']=dim_block
        block_dict[block]['p_value']=0

    for (block_s, block_t), flux in emp_bowtie_fluxes.items():
        flux_dict[(block_s, block_t)]['obs']=flux
        flux_dict[(block_s, block_t)]['p_value']=0

    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    n_workers = min(n_workers, n_runs)

    # Distribute runs as evenly as possible across workers
    base, remainder = divmod(n_runs, n_workers)
    chunks = [base + 1 if i < remainder else base for i in range(n_workers)]
    # Draw independent random seeds for each worker
    seeds = np.random.randint(0, 2**31, size=n_workers).tolist()
    task_args = [(sol, chunk, seed) for chunk, seed in zip(chunks, seeds)]

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = list(executor.map(_worker_chunk, task_args))

    # Aggregate results from all workers
    for blocks_list, fluxes_list in futures:
        for sim_bowtie_blocks in blocks_list:
            for block, dim_block in sim_bowtie_blocks.items():
                if dim_block >= block_dict[block]['obs']:
                    block_dict[block]['p_value']+=1/n_runs
        for sim_bowtie_fluxes in fluxes_list:
            for (block_s, block_t), flux in sim_bowtie_fluxes.items():
                if flux >= flux_dict[(block_s, block_t)]['obs']:
                    flux_dict[(block_s, block_t)]['p_value']+=1/n_runs

    return block_dict, flux_dict
    



def block_and_fluxes(weighted_el):
    '''
    This function takes a weighted edge list and extracts the bowtie structure, 
    returning the counts of nodes in each block and the fluxes among blocks.
    '''
    # extract the topological edge_list to feed bowtie
    topo_el = [(s, t) for s, t, w in weighted_el]
    bowtie_dict=edges2bowtie(topo_el)
    # measure empirical bowtie's block dimensions
    bowtie_counts=defaultdict(int)
    for node, block in bowtie_dict.items():
        bowtie_counts[block]+=1
    # measure empirical bowtie's fluxes among blocks
    bowtie_fluxes=defaultdict(int)
    for s, t, w in weighted_el:
        block_s=bowtie_dict[s]
        block_t=bowtie_dict[t]
        bowtie_fluxes[(block_s, block_t)]+=w
    return bowtie_counts, bowtie_fluxes

