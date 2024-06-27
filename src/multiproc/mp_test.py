#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 17:22:22 2024

@author: jp
"""

import numpy as np
import pandas as pd
import boost_histogram as bh

        
# Standard library imports:
import os
import pickle

# Package imports:
from mcpmeas.load_recentered_data import load_recentered_data

from multiproc.multiproc_cpu_test import main
from multiproc.hist_test import load, post_select_k_space, fill_single_hist


if __name__ == '__main__':
    
    
    
    # main(uj_vals, load_data, interval)
    # # Hist test:
    uj, recentered_data, k_sat_z_cutoff = load()
    ps_data = recentered_data.apply(post_select_k_space, axis=1, args=(k_sat_z_cutoff,))
    # Prepare data container:
    G2 = pd.Series(index=ps_data.index, dtype=object)
    # Calculate total correlations:
    # G2_list = [fill_single_hist(run, ps_data) for run in ps_data.index.to_numpy()]
    G2_list = main(ps_data.index.to_numpy(), fill_single_hist, ps_data, free_cores=0)
    
    # # Unpack multiprocessing results:    
    for run_idx, run_hist in zip(ps_data.index, G2_list):
        G2.at[run_idx] = run_hist
    # # Test:
    assert G2.apply(lambda x: x.sum()).sum() == 2002236.0  # 24390.0


