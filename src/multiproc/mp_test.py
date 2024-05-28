#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 17:22:22 2024

@author: jp
"""

import time
import numpy as np

from multiproc.multiproc_cpu import multiproc_cpu as mp

uj_vals = np.arange(200)

def load_data(uj):
    time.sleep(0.01)

mp(uj_vals, load_data, desc='Loading data', unit='Datasets')


