#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 15:16:31 2024

@author: jp
"""

from multiprocessing import Process, Queue
import time
import numpy as np
import enlighten
import psutil


def process_tasks(func, worker_queue, progress_queue, tasks, worker_idx, sub_array_sizes):
    """
    Simple child processor

    Sleeps for a random interval and pushes the current count onto the queue
    """

    for idx, i in enumerate(tasks):
        idx_offset_worker = sum(sub_array_sizes[:worker_idx - 1])

        progress_queue.put(idx)
        overall_idx = idx_offset_worker + idx
        worker_queue.put((overall_idx, func(i)))

def multiprocess_cores(manager, input_array, func, free_cores=0):
    """
    Process a random number of virtual tasks in subprocesses for the given number of cores
    """
    
    # Determine number of processing cores to be used:
    available_cores = psutil.cpu_count()
    try:
        assert free_cores < available_cores
    except AssertionError:
        raise ValueError(f'Number of free cores must be < {available_cores}.')
    
    nb_of_workers = available_cores - free_cores
    
    # Set padding for Core ID in progress bars:
    core_count_pad = len(str(nb_of_workers))
    
    # Split input array according to number of workers:
    sub_arrays = np.array_split(input_array, nb_of_workers)
    sub_array_sizes = [s.size for s in sub_arrays]
    
    # Set up progress bar:
    started = 0
    active = {}
    all_results = [None] * len(input_array)
    results = {}
    bar_format = u'{desc}{desc_pad}{percentage:3.0f}%|{bar}| ' + \
                 u'S:' + manager.term.yellow2(u'{count_0:{len_total}d}') + u' ' + \
                 u'F:' + manager.term.green3(u'{count_1:{len_total}d}') + u' ' + \
                 u'E:' + manager.term.red2(u'{count_2:{len_total}d}') + u' ' + \
                 u'[{elapsed}<{eta}, {rate:.2f}{unit_pad}{unit}/s]'

    pb_started = manager.counter(
        total=nb_of_workers, desc='Tasks:', unit='tasks', color='yellow2', bar_format=bar_format,
    )
    pb_finished = pb_started.add_subcounter('green3', all_fields=True)
    pb_error = pb_started.add_subcounter('red2', all_fields=True)



    worker_queue = Queue()
    
    


    # Loop until all cores finish
    while nb_of_workers > started or active:

        # If there are free cores and tasks left to run, start them
        if nb_of_workers > started and len(active) < nb_of_workers:
            progress_queue = Queue()
            tasks = sub_arrays[started]
            started += 1
            process = Process(target=process_tasks, name='Core %d' % started, args=(func, worker_queue, progress_queue, tasks, started, sub_array_sizes))
            counter = manager.counter(total=tasks.size, desc=f'  Core {started:>{core_count_pad}}:',
                                      unit='tasks', leave=False)
            process.start()
            pb_started.update()
            active[started] = (process, progress_queue, counter)
            results[started] = []

        # Iterate through active subprocesses
        for core in tuple(active.keys()):
            process, progress_queue, counter = active[core]
            single_res = results[core]
            alive = process.is_alive()

            # Latest count is the last one on the queue
            count = None
            while not progress_queue.empty():
                count = progress_queue.get()
                
            while not worker_queue.empty():
                single_res.append(worker_queue.get())

            # Update counter
            if count is not None:
                counter.update(count - counter.count)

            # Remove any finished subprocesses and update main progress bar
            if not alive:
                counter.close()
                print(f'Core {core:>{core_count_pad}} finished {counter.total} tasks')
                for a in single_res:
                    idx, i = a
                    all_results[idx] = i
                del active[core]

                # Check for failures
                if process.exitcode != 0:
                    print('ERROR: Receive exitcode %d while processing Core %d'
                          % (process.exitcode, core))
                    pb_error.update_from(pb_started)
                else:
                    pb_finished.update_from(pb_started)
                    
            # Reduce load:
            time.sleep(0.01)
            
    return all_results


def main(input_array, func):
    """
    Main function
    """

    with enlighten.get_manager() as manager:
        mp_res = multiprocess_cores(manager, input_array, func)
        
    return mp_res


if __name__ == '__main__':
    
    # input_array = np.array([None] * 10000, dtype=object)
    
    # for i in range(10000):
    #     input_array[i] = bh.Histogram()
        
    # input_array = np.array(input_array)
    
    # def func(i):
    #     time.sleep(0.001)
    #     return i + 1
    
    # res = main(input_array, func)

    # assert sum(res)==sum([i + 1 for i in input_array])
    
    # assert res==[i + 1 for i in input_array]
    
    from mcpmeas.load_recentered_data import load_recentered_data
    
    import os
    import pickle

    # Package imports:
    from mcpmeas.load_recentered_data import load_recentered_data
    
    # Constants:
    UJ_MAX = 22
    data_basepath = os.path.join(
        os.path.expanduser('~'), 'Data/MCP/conncorr/indiv_30_pct_rho_zero')

    def uj_from_filename(f):
        """String formatting helper function."""
        return float(f.split('uj')[1].split('_30_pct')[0].replace('p','.'))

    filenames = sorted(
        [filename for filename in os.walk(data_basepath)][0][2]
        )
    datapaths = sorted(
        [os.path.join(data_basepath, filename) for filename in filenames]
        )
    uj_vals = sorted(
        [uj_from_filename(filename) for filename in filenames]
        )

    # Load recentered data:
    main(
        datapaths,
        load_recentered_data
        )
    
    from mcpmeas.helper_functions import multiproc_list

