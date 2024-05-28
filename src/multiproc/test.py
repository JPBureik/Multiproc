#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 18:27:41 2024

@author: jp
"""

"""
Demo of Enlighten's features
"""

import os
import platform
import random
import time
import sys

import enlighten

# Hack so imports work regardless of how this gets called
# We do it this way so any enlighten path can be used
sys.path.insert(1, os.path.dirname(__file__))

# pylint: disable=wrong-import-order,import-error,wrong-import-position
from multicolored import run_tests, load  # noqa: E402
from multiple_logging import process_files, win_time_granularity  # noqa: E402
from prefixes import download  # noqa: E402


def initialize(manager, initials=15):
    """
    Simple progress bar example
    """

    # Simulated preparation
    pbar = manager.counter(total=initials, desc='Initializing:', unit='initials')
    for _ in range(initials):
        time.sleep(random.uniform(0.05, 0.25))  # Random processing time
        pbar.update()
    pbar.close()


def main():
    
    
    """
    Main function
    """

    manager = enlighten.get_manager()
    
    status = manager.status_bar(
        status_format=u'Running {filename}{fill}Stage: {task}{fill}{elapsed}',
        color='bold_underline_bright_white_on_lightslategray',
        justify=enlighten.Justify.CENTER,
        task='Initializing',
        filename=__file__.split('/')[-1],
        autorefresh=True
        )

    initialize(manager, 15)
    status.update(task='Loading')
    load(manager, 40)
    status.update(task='Testing')
    run_tests(manager, 20)
    status.update(task='Downloading')
    download(manager, 2.0 * 2 ** 20)
    status.update(task='File Processing')
    process_files(manager)


if __name__ == '__main__':

    if platform.system() == 'Windows':
        with win_time_granularity(1):
            main()
    else:
        main()