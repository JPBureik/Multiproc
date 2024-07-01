#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 16:24:28 2024

@author: jp

Progressbar class for the multiprocessing package.

May also be used as a stand-alone progress bar. The layout of the progress bar
can be modified in the corresponding methods of the class.
"""

# Standard library imports:
import enlighten
import inspect

# Package imports:
import multiproc

class Progressbar():
    r"""A progress bar for multiprocessing.
    
    The global manager and the status bar are initialized in the
    ``__init__.py`` file of the module.
    
    """
    
    def __init__(self, desc, total, unit):
        # Link global manager to instance:
        self.mngr = multiproc.mngr
        # Link global status bar to instance:
        self.sbar = multiproc.sbar
        # Update status bar with current task:
        self.sbar.update(task=desc)
        self.sbar.refresh()
        # Create progress bar:
        self.pbar = self.mngr.counter(total=total, desc=desc, unit=unit)


def init_statusbar(manager):
    r"""Initialize the global status bar.
    
    This function is called from the ``__init__.py`` file of the module.
    
    """
    
    return manager.status_bar(
        status_format=u'Running {fname}{fill}Stage: {task}{fill}{elapsed}',
        color='bold_underline_bright_white_on_lightslategray',
        justify=enlighten.Justify.CENTER,
        task='Initializing',
        fname=get_caller_script_name(),
        autorefresh=True
        )


def get_caller_script_name():
    """Return the file name of the outermost caller script.
    
    """
    # Get current frame:
    f = inspect.currentframe()
    # Get globals of outermost caller frame:
    om_caller_globals = inspect.getouterframes(f)[-1].frame.f_globals
    # Get subclass objects of BaseClass:
    return om_caller_globals['__file__'].split('/')[-1]
