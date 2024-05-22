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

    with enlighten.get_manager() as manager:
        status = manager.status_bar(status_format=u'Running %s{fill}Stage: {demo}{fill}{elapsed}' % __file__.split('/')[-1],
                                    color='bold_underline_bright_white_on_lightslategray',
                                    justify=enlighten.Justify.CENTER, demo='Initializing',
                                    autorefresh=True, min_delta=0.5)

        initialize(manager, 15)
        status.update(demo='Loading')
        load(manager, 40)
        status.update(demo='Testing')
        run_tests(manager, 20)
        status.update(demo='Downloading')
        download(manager, 2.0 * 2 ** 20)
        status.update(demo='File Processing')
        process_files(manager)


if __name__ == '__main__':

    if platform.system() == 'Windows':
        with win_time_granularity(1):
            main()
    else:
        main()