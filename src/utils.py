########################################################################
# Author(s):    D. Knowles
# Date:         15 Sep 2021
# Desc:         Common utility functions
########################################################################

import os
import sys

def prep_logs(log_name = ["default"]):
    """Create log directories if they don't yet exist.

    Parameters
    ----------
    log_name : list
        List of strings that are joined with underscores to create a new
        directory in the log directory

    """
    repo_dir = os.path.dirname(
               os.path.dirname(
               os.path.realpath(__file__)))

    log_dir = os.path.join(repo_dir,"log","_".join(log_name))

    # create log data directory if it doesn't yet exist
    if not os.path.isdir(log_dir):
        try:
            os.makedirs(log_dir)
        except OSError as e:
            print("e: ",e)
            sys.exit(1)

    return log_dir
