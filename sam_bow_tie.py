import os, sys, pickle
import datetime as dt
import numpy as np
from collections import defaultdict

from bowtie import edges2bowtie

def validate(weighted_el, sol, n_runs):
    emp_bowtie_blocks, emp_bowtie_fluxes=block_and_fluxes(weighted_el)
    block_dict=defaultdict(dict)
    flux_dict=defaultdict(dict)
    for block, dim_block in emp_bowtie_blocks.items():
        block_dict[block]['obs']=dim_block
        block_dict[block]['p_value']=0

    for (block_s, block_t), flux in emp_bowtie_fluxes.items():
        flux_dict[(block_s, block_t)]['obs']=flux
        flux_dict[(block_s, block_t)]['p_value']=0

    # Parallelizable loop to sample from the solution and compute the bowtie structure for each sample
    for i in range(n_runs):
        sampled_wel=sol.sample()
        sim_bowtie_blocks, sim_bowtie_fluxes=block_and_fluxes(sampled_wel)
        for block, dim_block in sim_bowtie_blocks.items():
            if dim_block >= block_dict[block]['obs']:
                block_dict[block]['p_value']+=1/n_runs
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

