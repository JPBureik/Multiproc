#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 16:48:12 2024

@author: jp

Initalize the global manager and status bar.
"""

# Standard library imports:
import enlighten

# Local package imports:
from .progressbar import init_statusbar

# Initialize progress bar manager on package level:
mngr = enlighten.get_manager()

# Initialize status bar:
sbar = init_statusbar(mngr)
