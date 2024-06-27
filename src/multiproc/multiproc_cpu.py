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

 #Package imports:
from multiproc import mngr, sbar
from multiproc.progressbar import Progressbar, get_subclass_objects


def multiproc_cpu(ary, func, *args, free_cores=None, desc='Processing',
                  unit='Iterations', **kwargs):
    r"""Multiprocessing on the CPU. Returns ``[func(i) for i in ary]`` in
    the original order.
    
    Parameters
    ----------
    ary: array_like
        Array over which parallelization is performed.
    func: function
        Function to be executed for each element of `ary`.
        The signature of `func` must be as follows:
            
            ``func(i, *args, **kwargs)``
            
        where
        
            * `i` is an element of `ary`
            * `args` are (optional) positional arguments of `func`
            * `kwargs` are (optional) keyword arguments of `func`
    
    free_cores: int, optional
        Number of threads on physical CPU cores to be left free. If `None`,
        Hyper Threading is used on all available cores.
    desc: string, optional
        Title for the multiprocessing progress bar.
    unit: string, optional
        Label for the multiprocessing progress bar.     
        
    Returns
    -------
    list
        The equivalent result of ``[func(i) for i in ary]`` in the
        original order.
        
    Raises
    -------
    ValueError
        If `free_cores` is not smaller than the total number of physical
        cores using Hyper Threading.
        
    Notes
    -----
    Parallelization is performed by splitting `ary` into chunks of
    approximate size ``len(ary) / (available_cores - free_cores)`` and
    assigning ``[func(i) for i in chunk]`` for the separate chunks to the
    different cores with Hyper Threading.
    
    """
    
    # Total number of iterations to be carried out:
    total = len(ary)

    # Initialize top level progress bar:
    pbar = Progressbar(desc, total, unit, func.__name__)
    
    # Set number of workers:
    nb_of_workers = set_nb_of_workers(free_cores)
    
    # Split input ary:
    sub_arrays = parallel_split(ary, nb_of_workers)
   
    
    
    
    
    # for element in ary:
    #     func(element)
    #     progressbar.pbar.update()
    # progressbar.pbar.close()


def set_nb_of_workers(free_cores):
    
    r"""Helper function to set the total number of threads to be used for
    multiprocessing.
    
    Parameters
    ----------
    free_cores: int
        Number of threads on physical CPU cores to be left free. If `None`,
        Hyper Threading is used on all available cores.
        
    Returns
    -------
    int
        Total number of threads across all physical CPU cores to be used for
        parallelization.
        
    Raises
    -------
    ValueError
        If `free_cores` is not smaller than the total number of physical
        cores using Hyper Threading.
    
    """
    
    # Determine number of available threads across all physical CPU cores:
    available_cores = psutil.cpu_count(logical=True)
    
    # Return the number of workers for parallelization:
    if not free_cores:
        return available_cores
    else:
        try:
            assert free_cores < available_cores
        except AssertionError:
            raise ValueError('Number of free cores must be < '
                             + f'{available_cores}.')
        return available_cores - free_cores


def parallel_split(ary, nb_of_workers):
    
    r"""Helper function to split the input ary into chunks according to
    the number of available workers.
    
    Parameters
    ----------
    ary: list or numpy.ndarray
        Variable over which parallelization is performed.
    nb_of_workers: int
        Total number of threads across all physical CPU cores to be used for
        parallelization.
        
    Returns
    -------
    list
        List of length `nb_of_workers`.
    
    """
    
    # Coerce to numpy array:
    if not isinstance(ary, np.ndarray):
        ary = np.array(ary)
    
    # Split input array according to the number of available workers:
    return np.array_split(ary, nb_of_workers)

if __name__ == '__main__':
    
    import time
    multiproc_cpu(np.ones(4), time.sleep, free_cores=None, desc='Processing',
                      unit='Iterations')
