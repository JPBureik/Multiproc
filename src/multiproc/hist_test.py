#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 09:43:42 2024

@author: jp
"""

import numpy as np
import pandas as pd
import boost_histogram as bh

from mcpmeas.load_recentered_data import load_recentered_data


CORREL_TYPE = 'k/-k'  # str, 'k/-k' or 'k/k'

# Set abs. values of measurement volume:
K_DEP_MIN = 0.3  # [k_d]
K_DEP_MAX = 0.7  # [k_d]

# Set abs. value of width of saturated slice to remove around z = 0:
K_SAT_Z = 0.05  # [k_d]

# Set histogram parameters:
PIXELSIZE_LONG = 0.012  # Bin size for histograms
NB_OF_BINS = 29  # odd int
K_RANGE = (NB_OF_BINS - 1) / 2 * PIXELSIZE_LONG  # Momentum range of the density cuts

# Create bins for histogram:
BIN_CENTERS = np.linspace(-K_RANGE, K_RANGE, NB_OF_BINS)
BIN_SIZES = np.diff(BIN_CENTERS[:2])[0] / 2
BIN_EDGES = BIN_CENTERS + BIN_SIZES
BIN_EDGES = np.insert(BIN_EDGES, 0, BIN_EDGES[0] - 2 * BIN_SIZES)

# Constants:
LATTICE_AXES = ('k_m45', 'k_h', 'k_p45')

def load():
    
    filename_load = '/home/jp/Data/MCP/conncorr/indiv_30_pct_rho_zero/uj20_30_pct.mat'
    
    uj, recentered_data, k_sat_z_cutoff = 20, load_recentered_data(filename_load, lattice_axes_only=False), 0.05
    
    return uj, recentered_data, k_sat_z_cutoff

#%% Momentum space post selection:

# Perform post-selection in momentum space:
def post_select_k_space(df_row, k_sat_z):

    # Get indices where |k| < K_DEP_MAX for all axes individually:
    within_upper_ax = [np.absolute(df_row[axis]) < K_DEP_MAX
                       for axis in LATTICE_AXES]
    # Get intersection of indices for all axes:
    within_upper = (within_upper_ax[0] & within_upper_ax[1]
                    & within_upper_ax[2])
    
    # Get indices where K_DEP_MIN < |k| for all axes individually:
    outside_lower_ax = [K_DEP_MIN < abs(df_row[axis])
                        for axis in LATTICE_AXES]
    
    # Get union of indices for all axes:
    outside_lower = (outside_lower_ax[0] | outside_lower_ax[1]
                     | outside_lower_ax[2])
    
    # Retain only momenta within measurement volume:
    idxs_to_keep = within_upper & outside_lower
    
    # Remove the z = 0 plane in which saturation effects occur:
    idxs_to_keep = np.logical_and(idxs_to_keep, abs(df_row['k_z']) >= k_sat_z)
    
    # Keep only post-select momenta along the lattice axes:
    return pd.Series({axis:
                      df_row.loc[axis][idxs_to_keep]
                      for axis in LATTICE_AXES})

#%% Total correlations:
    
# Loop over all shots and bin momentum sums into individual histograms:
def fill_single_hist(run, ps_data):
    
    # Prepare histogram:
    run_hist = bh.Histogram(
        bh.axis.Regular(len(BIN_CENTERS), BIN_EDGES[0], BIN_EDGES[-1]),
        bh.axis.Regular(len(BIN_CENTERS), BIN_EDGES[0], BIN_EDGES[-1]),
        bh.axis.Regular(len(BIN_CENTERS), BIN_EDGES[0], BIN_EDGES[-1])
        )
    
    # Prepare index mask:
    idx_mask = np.ones(ps_data['k_h'].loc[run].size, dtype=bool)
    
    momentum_m45 = ps_data['k_m45'].loc[run]
    momentum_h = ps_data['k_h'].loc[run]
    momentum_p45 = ps_data['k_p45'].loc[run]
    
    # Build momentum sums for all atoms in shot for all axes:
    for atom_idx, _ in enumerate(ps_data['k_h'].loc[run]):
    
        if CORREL_TYPE == 'k/-k':
            sum_m45 = momentum_m45 + momentum_m45[atom_idx]
            sum_H = momentum_h + momentum_h[atom_idx]
            sum_p45 = momentum_p45 + momentum_p45[atom_idx]
        elif CORREL_TYPE == 'k/k':
            sum_m45 = momentum_m45 - momentum_m45[atom_idx]
            sum_H = momentum_h - momentum_h[atom_idx]
            sum_p45 = momentum_p45 - momentum_p45[atom_idx]
            
        # Remove momentum sum of atom with itself:
        idx_mask[atom_idx] = False
        sum_m45 = sum_m45[idx_mask]
        sum_H = sum_H[idx_mask]
        sum_p45 = sum_p45[idx_mask]
        # Restore mask for use in next iteration:
        idx_mask[atom_idx] = True
        
        # Add to histogram:
        run_hist.fill(sum_m45, sum_p45, sum_H)      
        
    return run_hist
    