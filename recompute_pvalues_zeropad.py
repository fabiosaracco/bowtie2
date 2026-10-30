"""Recompute the stored bowtie p-values with the zero-padding fix of sam_bowtie.validate.

Reads every pvalues/*_pvalues_{blocks,fluxes}[_1|_decm].pkl and *_dwcm_pvalues_*.pkl that carries the realization vectors
('count_sample'), pads each vector with zeros up to the number of realizations (absent block/flux = 0 in that realization),
recomputes 'p_value' as validate() now does (block sizes: two-tailed; inter-block fluxes: right tail), and writes the result to pvalues_zp/ with the same file name.
Nothing in pvalues/ is modified. Old summary-only files (no vectors) are skipped.
"""
import glob, os, pickle, re
import numpy as np
from sam_bowtie import pvalue_block, pvalue_flux

HERE = os.path.dirname(os.path.abspath(__file__))
SRC, DST = os.path.join(HERE, 'pvalues'), os.path.join(HERE, 'pvalues_zp')


def main():
    os.makedirs(DST, exist_ok=True)
    files = sorted(glob.glob(os.path.join(SRC, '*_pvalues_*.pkl')))
    print(f"{'file':58s} {'entries':>7s} {'padded':>6s} {'p changed':>9s} {'by >0.05':>8s}")
    for f in files:
        name = os.path.basename(f)
        if 'lwcc' in name:
            continue
        d = pickle.load(open(f, 'rb'))
        if not any(isinstance(v, dict) and 'count_sample' in v for v in d.values()):
            continue                                              # old summary-only format
        fn = pvalue_flux if '_pvalues_fluxes' in name else pvalue_block
        n = max(len(v['count_sample']) for v in d.values())
        out = {}
        padded = chg = big = 0
        for k, v in d.items():
            s = v['count_sample']
            new = dict(v)
            new['count_sample'] = list(s) + [0] * (n - len(s))
            new['p_value'] = fn(s, v['obs'], n)
            out[k] = new
            padded += len(s) < n
            chg += abs(new['p_value'] - v['p_value']) > 1e-12
            big += abs(new['p_value'] - v['p_value']) > 0.05
        pickle.dump(out, open(os.path.join(DST, name), 'wb'))
        print(f"{name:58s} {len(d):7d} {padded:6d} {chg:9d} {big:8d}")


if __name__ == '__main__':
    main()
