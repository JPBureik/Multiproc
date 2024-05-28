#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 16:22:57 2024

@author: jp

Multiprocessing on the CPU.
"""

# Standard library imports:
import psutil
import numpy as np

from multiproc.progressbar import Progressbar

def set_nb_of_workers(free_cores):
    # Determine number of processing cores to be used:
    available_cores = psutil.cpu_count()
    
    if not free_cores:
        nb_of_workers = available_cores
    
    else:
    
        try:
            assert free_cores < available_cores
        except AssertionError:
            raise ValueError(f'Number of free cores must be < {available_cores}.')
        
        nb_of_workers = available_cores - free_cores
        
    return nb_of_workers

def split_input_iterable(iterable, nb_of_workers):
    sub_arrays = np.array_split(iterable, nb_of_workers)
    return sub_arrays
    
    
    
    
def multiproc_cpu(iterable, function, *args, desc='Processing',
                  free_cores=None, unit='Iterations', **kwargs):
    """Multiprocessing on the CPU."""
    
    total = len(iterable)

    # Initialize top level progress bar:
    progressbar = Progressbar(desc, total, unit, function.__name__)
    
    # Set number of workers:
    nb_of_workers = set_nb_of_workers(free_cores)
    
    # Split input iterable:
    sub_arrays = split_input_iterable(iterable, nb_of_workers)
   
    
    
    
    
    
    
    
    
    
    
    
    
    for element in iterable:
        function(element)
        progressbar.pbar.update()
    progressbar.pbar.close()
