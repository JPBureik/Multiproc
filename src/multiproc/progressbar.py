#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 16:24:28 2024

@author: jp

Progressbar for multiprocessing.
"""

# Standard library imports:
import enlighten
import inspect

# Package imports:
import multiproc

class Progressbar():
    
    r"""A progress bar for multiprocessing."""
    
    def __init__(self, desc, total, unit, fname):
        # Link module manager to instance:
        self.mngr = multiproc.mngr
        # Link status bar to instance:
        self.sbar = multiproc.sbar
        # Update status bar with current file name:
        self.sbar.update(fname=get_subclass_objects())
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


def get_subclass_objects():
    """Get all objects that extend BaseClass from global scope of the outermost
    caller frame."""
    # Get current frame:
    f = inspect.currentframe()
    # Get globals of outermost caller frame:
    om_caller_globals = inspect.getouterframes(f)[-1].frame.f_globals
    # Get subclass objects of BaseClass:
    return om_caller_globals['__file__'].split('/')[-1]