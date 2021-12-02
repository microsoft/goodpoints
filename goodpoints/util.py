"""Utility functions.
"""

def isnotebook():
    """Returns True if code is being executed interactively as a Jupyter notebook
    and False otherwise.
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter
        
def fprint(*args, verbose=True, **kwargs):
    """Flush print: If verbose is True, calls print with flush = True.
    """
    if verbose:
        print(*args, **kwargs, flush=True)
