#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 16:24:28 2024

@author: jp

Progressbar for multiprocessing.
"""

# Standard library imports:
import enlighten

# Package imports:
import multiproc

class Progressbar():
    
    def __init__(self, desc, total, unit, fname):
        # Link module manager to instance:
        self.mngr = multiproc.mngr
        # Link status bar to instance:
        self.sbar = multiproc.sbar
        # Update status bar with current file name:
        self.sbar.update(fname=fname)
        # Update status bar with current task:
        self.sbar.update(task=desc)
        self.sbar.refresh()
        # Create progress bar:
        self.pbar = self.mngr.counter(total=total, desc=desc, unit=unit)

def init_statusbar(manager):
    
    return manager.status_bar(
        status_format=u'Running {fname}{fill}Stage: {task}{fill}{elapsed}',
        color='bold_underline_bright_white_on_lightslategray',
        justify=enlighten.Justify.CENTER,
        task='Initializing',
        fname='__init__',
        autorefresh=True
        )
        
        