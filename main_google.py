########################################################################
# Author(s):    D. Knowles
# Date:         09 Aug 2021
# Desc:         runs the FDE across multiple traces
########################################################################

import os
import time
import numpy as np

from src.fde_google import EKF

def main(trace_list):

    total_time = 0.0

    ####################################################################
    # robustness test
    ####################################################################
    error_additions = [10.,20.,50.,100.,200.]
    fde_parameters = {
                      # threshold value for residual FDE
                      "residual" : np.linspace(0.89, 1., num = 7),

                      # threshold value for EDM FDE
                      "edm" : np.logspace(0, 3, num = 13),

                      # threshold value for solution separation FDE
                      "solution" : [50,75],
                     }

    log_names = ["google","fde"]

    for ii, trace in enumerate(trace_list):
        trace_name, phone_type = trace
        time0 = time.time()

        ekf = EKF(trace_name, phone_type, fde_parameters,
                  log_names.copy(), True)
        test_path = ekf.run()

        trace_time = time.time() - time0
        total_time += trace_time
        print("trace FDE analysis ",ii+1,"/",len(trace_list),
              " took ", round(trace_time,3), " sec.")
        print("total time of ", round(total_time/60.,2), " minutes.")

    ####################################################################
    # measurement count timing tests using google dataset
    ####################################################################
    fde_parameters = {
                      # threshold value for residual FDE
                      "residual" : [0.9440608762859234],

                      # threshold value for EDM FDE
                      "edm" : [17.78279410038923],

                      # threshold value for solution separation FDE
                      "solution" : [50.],
                     }
    log_names = ["google","timing"]

    for ii, trace in enumerate(trace_list):
        trace_name, phone_type = trace
        time0 = time.time()

        ekf = EKF(trace_name, phone_type, fde_parameters,
                  log_names.copy(), True)
        test_path = ekf.run()

        trace_time = time.time() - time0
        total_time += trace_time
        print("trace timing analysis ",ii+1,"/",len(trace_list),
              " took ", round(trace_time,3), " sec.")
        print("total time of ", round(total_time/60.,2), " minutes.")

if __name__ == "__main__":

    trace_list = [
                  ('2020-05-14-US-MTV-1', 'Pixel4'),
    ]

    main(trace_list)
