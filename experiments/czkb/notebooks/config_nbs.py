""" 
Adding additional paths from the workspace to be imported
with every notebook, so as not to include it in every notebok.
Tested using run.sh with the export function, but DBX does not
allow for sh. files to be run w/o the notebook. So this is the 
workaround.
"""

# Set up local Python source code as modules
import sys

# Define project specific paths
my_paths = [
    "/Workspace/Users/sg7cb@s-mxs.net/hg-ds-evals/",
    "/Workspace/Users/sg7cb@s-mxs.net/hg-ds-evals/experiments/czkb/",
]

def add_local_paths():
    for path in my_paths:
        if path not in sys.path:
            sys.path.append(path)
            print(f'Path "{path}" added!')
