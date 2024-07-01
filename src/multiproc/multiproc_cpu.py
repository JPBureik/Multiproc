#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 16:22:57 2024

@author: jp

Multiprocessing on the CPU.

The multiprocessing part of this package is kept functional instead of object-
oriented in order to facilitate state-independent parallelization.
"""

# Standard library imports:
import psutil
import numpy as np
import pandas as pd
from multiprocessing import Process, Queue

# Package imports:
from multiproc import mngr, sbar
from multiproc.progressbar import Progressbar


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
    progressbar = Progressbar(desc, total, unit)
    
    # Set number of workers:
    nb_of_workers = _set_nb_of_workers(free_cores)
    
    # Split input ary:
    sub_arys = _parallel_split(ary, nb_of_workers)
    
    # Run multiprocessing:
    return _mp_loop(sub_arys, func, nb_of_workers, total, progressbar, *args,
                    **kwargs)


def _set_nb_of_workers(free_cores):
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


def _parallel_split(ary, nb_of_workers):
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


def _mp_loop(sub_arys, func, nb_of_workers, total, progressbar, *args,
             **kwargs):
    r"""Multiprocessing loop.
    
    Assign tasks to workers in parallel and return the results in an ordered
    list.
    
    Parameters
    ----------
    sub_arys: list
        List containing the split of `ary` into `nb_of_workers`.
    func: function
        Function to be executed for each element of `ary`.
        The signature of `func` must be as follows:
            
            ``func(i, *args, **kwargs)``
            
        where
        
            * `i` is an element of `ary`
            * `args` are (optional) positional arguments of `func`
            * `kwargs` are (optional) keyword arguments of `func`
    
    nb_of_workers: int
        Total number of threads across all physical CPU cores to be used for
        parallelization.
    total: int
        Total number of iterations to be carried out.
    progressbar: multiproc.Progressbar
        Top level progress bar to keep track of the multiprocessing progress.
        
    Returns
    -------
    list
        The equivalent result of ``[func(i) for i in ary]`` in the
        original order.
    
    """
    
    # Initialize data container for results:
    mp_res = [[] for _ in range(total)]
    
    # Initialize queue to gather results returned by different workers:
    worker_queue = Queue()
    
    # DataFrame to keep track of the active workers:
    active_processes = pd.DataFrame(
        index=range(nb_of_workers),
        columns=['Process', 'Progress Queue', 'Counter', 'Active'],
        dtype=object
        )
    
    # Loop until all cores finish:
    while active_processes['Active'].sum() > 0:
        
        # Initialize new worker if there are free cores and tasks left to run:
        if active_processes['Active'].sum() < nb_of_workers:
            active_processes = _start_process(sub_arys, func, active_processes,
                                              worker_queue, *args, **kwargs)
            
        # Check status of started processes:
        _poll_active_processes(active_processes)
        
        # Reduce load of continuous polling:
        time.sleep(0.001)  # Define as top-level macro for easy access?
    
    return mp_res


def _start_process(active_processes, worker_queue, sub_arys, func, *args,
                   **kwargs):
    r"""Start a process from available tasks and return the updated DataFrame
    tracking the active processes.
    
    Parameters
    ----------
    sub_arys: list
        List containing the split of `ary` into `nb_of_workers`.
    func: function
        Function to be executed for each element of `ary`.
        The signature of `func` must be as follows:
            
            ``func(i, *args, **kwargs)``
            
        where
        
            * `i` is an element of `ary`
            * `args` are (optional) positional arguments of `func`
            * `kwargs` are (optional) keyword arguments of `func`
    
    active_processes: pandas.DataFrame, dtype: object
        DataFrame containing the created processes.
            Index: RangeIndex
                Indices of the processes.
            Columns: Index, dtype: object
                Process instance, progress queue, counter and activity status.
    worker_queue: multiprocessing.Queue
        Queue that gathers results returned by different workers.
    
    Returns
    -------
    active_processes: pandas.DataFrame, dtype: object
        DataFrame containing the created processes.
            Index: RangeIndex
                Indices of the processes.
            Columns: Index, dtype: object
                Process instance, progress queue, counter and activity status.
    
    """
    
    # Queue to track progress of single worker:
    single_progress_queue = Queue()
    
    # Sizes of tasks for each worker for progress bar:
    sub_ary_sizes = [sub_ary.size for sub_ary in sub_arys]
    
    # Index of next worker to be initialized:
    new_worker_idx = active_processes[
        active_processes['Active']==False].index[0]
    
    # Get task to assign to new worker:
    task = sub_arys[new_worker_idx]
    
    # Assign task to new process:
    process = Process(
        target=_process_task,
        name=f'Core {new_worker_idx}',
        args=(
            func,
            single_progress_queue,
            worker_queue,
            task,
            new_worker_idx,
            sub_ary_sizes,
            *args),
        kwargs=kwargs
        )
    
    # Start new process:
    process.start()
    
    # Add to DataFrame of active processes:
    active_processes.loc[new_worker_idx] = pd.Series({
        'Process': process,
        'Progress Queue': single_progress_queue,
        'Counter': None,
        'Active': True
        })
    
    return active_processes
    
    
def _process_task(func, progress_queue, worker_queue, task,
                   new_worker_idx, sub_ary_sizes, *args, **kwargs):
    r"""Function call for individual worker to process single chunk of split
    array.
    
    Parameters
    ----------
    
    
    """
    
    
    
    return

def _poll_active_processes(active_processes):
    
    for active_idx in active_processes[active_processes['Active']].index:
        
        print(active_idx)
    
    return

if __name__ == '__main__':
    
    import time
    multiproc_cpu(np.ones(4), time.sleep, free_cores=None, desc='Processing',
                      unit='Iterations')
