"""CPU multiprocessing with integrated progress tracking.

This module provides a simple functional API for parallel processing of
iterable data with real-time progress visualization using enlighten.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from multiprocessing import Process, Queue
from typing import Any, TypeVar

import enlighten  # type: ignore[import-untyped]
import numpy as np
import psutil
from numpy.typing import NDArray

T = TypeVar("T")


def multiproc_cpu(
    ary: Sequence[T] | NDArray[Any],
    func: Callable[..., Any],
    *args: Any,
    free_cores: int | None = None,
    desc: str = "Processing",
    unit: str = "items",
    **kwargs: Any,
) -> list[Any]:
    """Execute a function in parallel across multiple CPU cores.

    Applies `func` to each element of `ary` using multiprocessing with
    integrated progress tracking. Returns results in the original order.

    Parameters
    ----------
    ary : Sequence or NDArray
        Input data to process. Each element is passed to `func`.
    func : Callable
        Function to apply to each element. Signature: ``func(item, *args, **kwargs)``.
    *args : Any
        Positional arguments passed to `func` after the item.
    free_cores : int, optional
        Number of CPU threads to leave unused. If None, uses all available threads.
    desc : str, default "Processing"
        Description shown in the progress bar.
    unit : str, default "items"
        Unit label for the progress bar.
    **kwargs : Any
        Keyword arguments passed to `func`.

    Returns
    -------
    list
        Results in the same order as the input array: ``[func(item) for item in ary]``.

    Raises
    ------
    ValueError
        If `free_cores` >= total available threads.

    Examples
    --------
    >>> def square(x):
    ...     return x ** 2
    >>> results = multiproc_cpu([1, 2, 3, 4], square)
    >>> results
    [1, 4, 9, 16]

    """
    total = len(ary)
    if total == 0:
        return []

    nb_of_workers = _set_nb_of_workers(free_cores)
    sub_arys = _parallel_split(ary, nb_of_workers)
    sub_ary_sizes = [len(sub_ary) for sub_ary in sub_arys]

    with enlighten.get_manager() as manager:
        return _mp_loop(
            manager,
            sub_arys,
            sub_ary_sizes,
            func,
            nb_of_workers,
            total,
            desc,
            unit,
            *args,
            **kwargs,
        )


def _set_nb_of_workers(free_cores: int | None) -> int:
    """Determine the number of worker processes to spawn.

    Parameters
    ----------
    free_cores : int or None
        Number of threads to leave unused. None uses all threads.

    Returns
    -------
    int
        Number of workers to use.

    Raises
    ------
    ValueError
        If free_cores >= available threads.

    """
    available_cores: int = psutil.cpu_count(logical=True) or 1

    if free_cores is None:
        return available_cores

    if free_cores >= available_cores:
        raise ValueError(
            f"free_cores ({free_cores}) must be less than "
            f"available threads ({available_cores})."
        )
    return available_cores - free_cores


def _parallel_split(
    ary: Sequence[T] | NDArray[Any], nb_of_workers: int
) -> list[NDArray[Any]]:
    """Split the input array into chunks for parallel processing.

    Parameters
    ----------
    ary : Sequence or NDArray
        Input data to split.
    nb_of_workers : int
        Number of chunks to create.

    Returns
    -------
    list of NDArray
        List of array chunks, one per worker.

    """
    if not isinstance(ary, np.ndarray):
        ary = np.array(ary, dtype=object)
    return list(np.array_split(ary, nb_of_workers))


def _mp_loop(
    manager: enlighten.Manager,
    sub_arys: list[NDArray[Any]],
    sub_ary_sizes: list[int],
    func: Callable[..., Any],
    nb_of_workers: int,
    total: int,
    desc: str,
    unit: str,
    *args: Any,
    **kwargs: Any,
) -> list[Any]:
    """Main multiprocessing loop that coordinates workers and collects results.

    Parameters
    ----------
    manager : enlighten.Manager
        Progress bar manager.
    sub_arys : list of NDArray
        Split input arrays, one per worker.
    sub_ary_sizes : list of int
        Size of each sub-array.
    func : Callable
        Function to apply to each element.
    nb_of_workers : int
        Number of parallel workers.
    total : int
        Total number of items to process.
    desc : str
        Progress bar description.
    unit : str
        Progress bar unit label.
    *args, **kwargs
        Arguments passed to func.

    Returns
    -------
    list
        Results in original order.

    """
    results: list[Any] = [None] * total
    worker_queue: Queue[tuple[int, Any]] = Queue()

    core_count_pad = len(str(nb_of_workers))

    main_bar = manager.counter(
        total=total,
        desc=desc,
        unit=unit,
        color="bold_bright_white_on_lightslategray",
    )

    active: dict[int, tuple[Process, Queue[int], enlighten.Counter]] = {}
    started = 0

    while started < nb_of_workers or active:
        if started < nb_of_workers and len(active) < nb_of_workers:
            progress_queue: Queue[int] = Queue()
            tasks = sub_arys[started]
            worker_idx = started
            started += 1

            process = Process(
                target=_process_task,
                name=f"Core {worker_idx}",
                args=(
                    func,
                    progress_queue,
                    worker_queue,
                    tasks,
                    worker_idx,
                    sub_ary_sizes,
                    *args,
                ),
                kwargs=kwargs,
            )

            counter = manager.counter(
                total=len(tasks),
                desc=f"  Core {worker_idx:>{core_count_pad}}:",
                unit=unit,
                leave=False,
            )

            process.start()
            active[worker_idx] = (process, progress_queue, counter)

        for worker_idx in list(active.keys()):
            process, progress_queue, counter = active[worker_idx]

            count = None
            while not progress_queue.empty():
                count = progress_queue.get()

            while not worker_queue.empty():
                idx, result = worker_queue.get()
                results[idx] = result
                main_bar.update()

            if count is not None:
                new_count = count + 1
                if new_count > counter.count:
                    counter.update(new_count - counter.count)

            if not process.is_alive():
                while not progress_queue.empty():
                    progress_queue.get()
                while not worker_queue.empty():
                    idx, result = worker_queue.get()
                    results[idx] = result
                    main_bar.update()

                counter.close()
                del active[worker_idx]

        time.sleep(0.001)

    main_bar.close()
    return results


def _process_task(
    func: Callable[..., Any],
    progress_queue: Queue[int],
    worker_queue: Queue[tuple[int, Any]],
    tasks: NDArray[Any],
    worker_idx: int,
    sub_ary_sizes: list[int],
    *args: Any,
    **kwargs: Any,
) -> None:
    """Worker function that processes a chunk of the input array.

    Parameters
    ----------
    func : Callable
        Function to apply to each item.
    progress_queue : Queue
        Queue for reporting progress to the main process.
    worker_queue : Queue
        Queue for returning results to the main process.
    tasks : NDArray
        Items this worker should process.
    worker_idx : int
        Index of this worker (for calculating global indices).
    sub_ary_sizes : list of int
        Sizes of all sub-arrays (for calculating global indices).
    *args, **kwargs
        Arguments passed to func.

    """
    idx_offset = sum(sub_ary_sizes[:worker_idx])

    for idx, item in enumerate(tasks):
        progress_queue.put(idx)
        overall_idx = idx_offset + idx
        result = func(item, *args, **kwargs)
        worker_queue.put((overall_idx, result))
