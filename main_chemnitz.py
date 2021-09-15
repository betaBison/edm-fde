########################################################################
# Author(s):    D. Knowles
# Date:         09 Aug 2021
# Desc:         runs the FDE on the Chemnitz dataset
########################################################################

import os
import time
import numpy as np

from src.fde_chemnitz import EKF

def main():

    total_time = 0.0

    ####################################################################
    # parameter sensitivity test
    ####################################################################
    error_addition = 50.
    fde_parameters = {
                      # threshold value for residual FDE
                      "residual" : np.logspace(-2, 2, num = 41),

                      # threshold value for EDM FDE
                      "edm" : np.logspace(-2, 2, num = 41),

                      # threshold value for solution separation FDE
                      "solution" : np.logspace(-2, 2, num = 21),
                     }
    fault_hypothesis = 1

    log_names = ["chemnitz","fde","detailed"]

    time0 = time.time()
    ekf = EKF(error_addition, fde_parameters, fault_hypothesis,
              log_names.copy(), True)
    test_path = ekf.run()

    trace_time = time.time() - time0
    total_time += trace_time
    print("parameter analysis took ", round(trace_time,3), " sec.")
    print("total time of ", round(total_time/60.,2), " minutes.")

    ####################################################################
    # chemnitz FDE tests
    ####################################################################
    error_additions = [10.,20.,50.,100.,200.]
    fde_parameters = {
                      # threshold value for residual FDE
                      "residual" : np.logspace(-2, 2, num = 21),

                      # threshold value for EDM FDE
                      "edm" : np.logspace(-2, 2, num = 21),

                      # threshold value for solution separation FDE
                      "solution" : np.logspace(-2, 2, num = 6),
                     }
    fault_hypothesis = 1

    log_names = ["chemnitz","fde"]

    for ii, error_addition in enumerate(error_additions):
        time0 = time.time()
        ekf = EKF(error_addition, fde_parameters, fault_hypothesis,
                  log_names.copy(), True)
        test_path = ekf.run()

        trace_time = time.time() - time0
        total_time += trace_time
        print("fde analysis ",ii+1,"/",len(error_additions),
              " took ", round(trace_time,3), " sec.")
        print("total time of ", round(total_time/60.,2), " minutes.")

    ####################################################################
    # fault hypothesis timing tests using TU Chemnitz dataset
    ####################################################################
    error_addition = 50.
    fde_parameters = {
                      # threshold value for residual FDE
                      "residual" : [15.848931924611142],

                      # threshold value for EDM FDE
                      "edm" : [1.584893192461114],

                      # threshold value for solution separation FDE
                      "solution" : [15.848931924611142],
                     }
    log_names = ["chemnitz","timing"]
    fault_hypotheses = [1, 2, 3]

    for ii, fault_hypothesis in enumerate(fault_hypotheses):
        time0 = time.time()
        ekf = EKF(error_addition, fde_parameters, fault_hypothesis,
                  log_names.copy(), True)
        test_path = ekf.run()

        trace_time = time.time() - time0
        total_time += trace_time
        print("timing analysis ",ii+1,"/",len(fault_hypotheses),
              " took ", round(trace_time,3), " sec.")
        print("total time of ", round(total_time/60.,2), " minutes.")

if __name__ == "__main__":
    main()
