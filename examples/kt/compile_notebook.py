"""Code to compile a Jupyter notebook into a Python script

Usage: python compile_notebook.py yournotebook.ipynb
"""
import subprocess, sys

from goodpoints.util import fprint  # for printing while flushing buffer
from goodpoints.tictoc import tic, toc # for timing blocks of code
def compile_notebook(ntbk):
    """Converts a jupyter notebook to a Python script

    Args:
      ntbk - file name of jupyter notebook ending in .ipynb
    """
    tic()
    # Convert jupyter notebook to script and remove extraneous folders generated 
    # by nbconvert
    import os
    output = os.path.abspath(ntbk).replace(".ipynb","")
    subprocess.call(f"jupyter nbconvert --to script {ntbk}; "+
                    "rm -rf nbconvert; mv \~ deleteme; rm -rf deleteme",
                    shell=True)
    toc()
    
# When called as a script, call compile_notebook on command line argument
def main():
    compile_notebook(sys.argv[1])
    
if __name__ == "__main__":
   main()
