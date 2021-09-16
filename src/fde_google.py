########################################################################
# Author(s):    D. Knowles
# Date:         11 Aug 2021
# Desc:         runs FDE methods with EKF implementation on google
#               android dataset
########################################################################

import os
import csv
import time
import random
import numpy as np
import progress.bar
import pandas as pd

from src.utils import prep_logs
import lib.coordinates as coord
from lib.android import make_gnss_dataframe
from src.edm_fde import edm_from_satellites_ranges, edm_fde

####################################################################
# PARAMETERS
####################################################################

# how many measurement rows to calculate
# -1 means keep all measurements
MEASUREMENTS_HEAD = -1

# ground truth fault indicator. Residuals with absolute value higher
# than this will be marked faulty.
FAULT_THRESHOLD = 100

# global parameters for the FDE methods
FDE_PARAMETERS = {
                    # this parameter doesn't actually do anything but
                    # makes the logic further down easier
                    "base" : [0],

                    # this parameter doesn't actually do anything but
                    # makes the logic further down easier
                    "truth" : [0],
                  }



####################################################################
# END PARAMETERS
####################################################################

class EKF():
    def __init__(self, trace_name, phone_type, fde_parameters,
                 log_names, verbose = False):
        """Exteded Kalman filter for gnss measurements.

        Parameters
        ----------
        trace_name : string
            Provided name for trace, e.g., "2020-05-14-US-MTV-1".
        phone_type : string
            Type of phone used to record data, e.g., "Pixel4",
            "Pixel4XL", "Pixel4XLModded", "Mi8".
        fde_parameters : dict
            FDE parameters to add to the FDE_PARAMETERS dictionary
        log_names : list
            Strings that get added to the name log
        verbose : bool
            whether or not to include extra outputs

        """

        print("Analyzing trace ", trace_name, phone_type, "...")

        # append added FDE parameters to FDE_PARAMETERS
        for key, value in fde_parameters.items():
            FDE_PARAMETERS[key] = value

        # update log name
        self.log_names = log_names
        self.log_names.append(str(int(MEASUREMENTS_HEAD)))
        self.log_names.append(str(int(FAULT_THRESHOLD)))

        repo_dir = os.path.dirname(
                   os.path.dirname(
                   os.path.realpath(__file__)))

        data_path = os.path.join(repo_dir,"data","google","train")

        # load measurements
        input_filepath = os.path.join(data_path, trace_name, phone_type,
                                      phone_type + "_derived.csv")
        measurements = pd.read_csv(input_filepath)

        # crop measurements according to MEASUREMENTS_HEAD parameter
        if MEASUREMENTS_HEAD != -1:
            measurements = measurements.head(MEASUREMENTS_HEAD)

        # calculate corrected psuedorange
        measurements["correctedPrM"] = measurements["rawPrM"] \
                                     - measurements["isrbM"] \
                                     - measurements["ionoDelayM"] \
                                     - measurements["tropoDelayM"]
        # add SvName
        measurements["SvName"] = measurements["constellationType"].replace(
                                 [1,3,4,5,6],["G","R","Q","B","E"]) \
                               + measurements["svid"].astype(str)

        # load ground truth
        gt_filepath = os.path.join(data_path, trace_name, phone_type,
                                      "ground_truth.csv")
        gt = pd.read_csv(gt_filepath)

        meas_times = []

        ####################################################################
        # MULTIPLE FIXES FROM DATA HOSTS
        # code copied from
        # https://www.kaggle.com/gymf123/tips-notes-from-the-competition-hosts
        ####################################################################

        # load raw measurements
        input_filepath = os.path.join(data_path, trace_name, phone_type,
                                      phone_type + "_GnssLog.txt")
        df_raw, android_fixes = make_gnss_dataframe(input_filepath, True)

        # Create a new column in df_raw that corresponds to df_derived['MillisSinceGpsEpoch']
        df_raw['millisSinceGpsEpoch'] = np.floor( (df_raw['TimeNanos'] - df_raw['FullBiasNanos']) / 1000000.0).astype(int)

        # Change each value in df_derived['MillisSinceGpsEpoch'] to be the prior epoch.
        raw_timestamps = df_raw['millisSinceGpsEpoch'].unique()
        derived_timestamps = measurements['millisSinceGpsEpoch'].unique()

        # The timestamps in derived are one epoch ahead. We need to map each epoch
        # in derived to the prior one (in Raw).
        indexes = np.searchsorted(raw_timestamps, derived_timestamps)
        from_t_to_fix_derived = dict(zip(derived_timestamps, raw_timestamps[indexes-1]))
        measurements['millisSinceGpsEpoch'] = np.array(list(map(lambda v: from_t_to_fix_derived[v], measurements['millisSinceGpsEpoch'])))

        # remove 61m bias in ground truth
        gt.loc[:,"heightAboveWgs84EllipsoidM"] -= 61.

        # remove duplicated signals that produce outliers
        delta_millis = measurements['millisSinceGpsEpoch'] - measurements['receivedSvTimeInGpsNanos'] / 1e6

        where_good_signals = (delta_millis > 0) & (delta_millis < 300)

        measurements = measurements[where_good_signals].copy()

        measurements.reset_index(drop=True,inplace=True)

        ####################################################################
        # END OF FIXES
        ####################################################################

        gt_lla = np.vstack((gt["latDeg"],gt["lngDeg"],
                            gt["heightAboveWgs84EllipsoidM"])).T
        gt_ecef = coord.geodetic2ecef(gt_lla)
        gt["ecefxM"] = gt_ecef[:,0]
        gt["ecefyM"] = gt_ecef[:,1]
        gt["ecefzM"] = gt_ecef[:,2]

        # number of epochs
        self.epochs = len(measurements['millisSinceGpsEpoch'].unique())

        initial_df = measurements[measurements["millisSinceGpsEpoch"] ==
                                  measurements.loc[measurements.first_valid_index(),"millisSinceGpsEpoch"]]
        initial_df.reset_index(drop=True,inplace=True)
        wls_x = np.zeros((1,3))
        wls_rb = 0.0
        wls_x, wls_rb = self.least_squares(wls_x,
                                          wls_rb,
                                          initial_df,False)

        # set initial positions
        x0 = wls_x[0,0]
        y0 = wls_x[0,1]
        z0 = wls_x[0,2]
        b0 = wls_rb

        # store measurements and gt in class variables
        self.gt = gt
        self.measurements = measurements
        self.trace_name = trace_name
        self.phone_type = phone_type

        # initialize state vector [ x, y, z ]
        self.mu = {}
        self.mu_history = {}
        # self.wls_history = {}
        for method in list(FDE_PARAMETERS.keys()):
            self.mu[method] = {}
            self.mu_history[method] = {}
            for param in FDE_PARAMETERS[method]:
                self.mu[method][param] = np.array([[x0,y0,z0,b0]]).T
                self.mu_history[method][param] = self.mu[method][param].copy()
        first_key = list(FDE_PARAMETERS.keys())[0]
        self.gt_history = self.mu[first_key][0].copy()
        self.mu_n = self.mu[first_key][0].shape[0]

        # initialize covariance matrix
        self.P = {}
        self.P_history = {}
        for method in list(FDE_PARAMETERS.keys()):
            self.P[method] = {}
            self.P_history[method] = {}
            for param in FDE_PARAMETERS[method]:
                self.P[method][param] = np.eye(self.mu_n)
                self.P_history[method][param] = np.zeros((1,self.mu_n,self.mu_n))
                self.P_history[method][param][0,:,:] = self.P[method][param]

        # computation time analysis
        self.timing_history = {}
        for method in list(FDE_PARAMETERS.keys()):
            self.timing_history[method] = {}

        # accuracy metrics
        self.fde_accuracy = {}
        accuracy_types = ["tp","tn","fn","fp"]

        for method in list(FDE_PARAMETERS.keys()):
            self.fde_accuracy[method] = {}
            for param in FDE_PARAMETERS[method]:
                self.fde_accuracy[method][param] = {}
                for accuracy_type in accuracy_types:
                    self.fde_accuracy[method][param][accuracy_type] = 0

    def predict_simple(self, method, param):
        """EKF predict step.

        method : string
            FDE method name
        param : float
            FDE method's thresholding parameter

        """
        # build state transition model matrix
        F = np.eye(self.mu_n)

        # update predicted state
        self.mu[method][param] = F.dot(self.mu[method][param])

        # build process noise matrix
        self.Q_cov = 0.1
        Q = np.eye(self.mu_n) * self.Q_cov
        Q[-1,-1] = 10.

        # propagate covariance matrix
        self.P[method][param] = F.dot(self.P[method][param]).dot(F.T) + Q

    def update_gnss(self, data, method, param):
        """EKF update step.

        data : pd.DataFrame
            DataFrame that contains information from the measurements
        method : string
            FDE method name
        param : float
            FDE method's thresholding parameter

        """
        num_sats = len(data)

        sat_pos = np.hstack((data["xSatPosM"].to_numpy().reshape(-1,1),
                             data["ySatPosM"].to_numpy().reshape(-1,1),
                             data["zSatPosM"].to_numpy().reshape(-1,1)))
        mu_pos = np.tile(self.mu[method][param][:3].T,(sat_pos.shape[0],1))

        gt_psuedoranges = np.linalg.norm(mu_pos - sat_pos, axis = 1)

        H = np.ones((num_sats,self.mu_n))
        H[:,:3] = np.divide(mu_pos - sat_pos,gt_psuedoranges.reshape(-1,1))

        R = np.diag(1./data["rawPrUncM"]**2)

        yt = data["correctedPrM"].to_numpy() \
           + data["satClkBiasM"].to_numpy() \
           - gt_psuedoranges \
           - self.mu[method][param][3]

        yt = yt.reshape(-1,1)

        Kt = self.P[method][param].dot(H.T).dot(np.linalg.inv(R + H.dot(self.P[method][param]).dot(H.T)))

        self.mu[method][param] = self.mu[method][param].reshape((-1,1)) + Kt.dot(yt)
        self.P[method][param] = (np.eye(self.mu_n)-Kt.dot(H)).dot(self.P[method][param])

    def run(self):
        """Run FDE methods.

        Returns
        -------
        log_dir : string
            filepath to logged data

        """
        # setup progress bar
        bar = progress.bar.IncrementalBar('Progress:', max=self.epochs)

        # Ephemeris calculation and Newton-Raphson solution
        nr_lla = []
        nr_ecef = []
        meas_times = []
        num_sats = []
        cn0 = []
        errors_added = 0

        for _,data in self.measurements.groupby("millisSinceGpsEpoch",as_index=False):

            data.reset_index(drop=True,inplace=True)

            # need at least six satellites for exclusion
            if len(data) < 6:
                bar.next() # progress bar
                continue

            # calculate ground truth fault status
            epoch_gt, mu_gt = self.calc_gt(data.copy(),self.gt.copy())

            gt_indexes = set(epoch_gt[epoch_gt["fault"] == False].index)
            faults_exist = len(epoch_gt[epoch_gt["fault"] == True]) > 0

            # compute EKF updates
            for method in list(FDE_PARAMETERS.keys()):
                for param in FDE_PARAMETERS[method]:
                    self.predict_simple(method, param)

                    input_data = data.copy()

                    time0 = time.time()
                    # run each FDE method
                    if method == "base":
                        new_data = input_data
                    elif method == "truth":
                        new_data = epoch_gt[epoch_gt["fault"] == False].copy()
                    elif method == "edm":
                        new_data = self.run_edm_fde(input_data, param, False)
                    elif method == "residual":
                        new_data = self.run_residual_fde(input_data, param, False)
                    elif method == "solution":
                        new_data = self.run_solution_fde(input_data, param, False)

                    time1 = time.time()

                    if len(epoch_gt[epoch_gt["fault"] == True]) <= 1:
                        if len(data) not in self.timing_history[method]:
                            self.timing_history[method][len(data)] = [time1-time0]
                        else:
                            self.timing_history[method][len(data)].append(time1-time0)

                    self.update_gnss(new_data, method, param)

                    self.mu_history[method][param] = np.hstack((self.mu_history[method][param],
                                                         self.mu[method][param]))
                    self.P_history[method][param] = np.concatenate((self.P_history[method][param],
                                             np.expand_dims(self.P[method][param],0)))

                    # accuracy metrics
                    method_indexes = set(new_data.index)
                    tn_e = len(gt_indexes & method_indexes)
                    fp_e = len(gt_indexes - method_indexes)
                    fn_e = len(method_indexes - gt_indexes)
                    tp_e = len(epoch_gt) - tn_e - fp_e - fn_e

                    self.fde_accuracy[method][param]["tp"] += tp_e
                    self.fde_accuracy[method][param]["tn"] += tn_e
                    self.fde_accuracy[method][param]["fn"] += fn_e
                    self.fde_accuracy[method][param]["fp"] += fp_e

            # add values to history
            self.gt_history = np.hstack((self.gt_history,
                                         mu_gt.reshape(-1,1)))

            bar.next() # progress bar

        bar.finish() # end progress bar

        self.log_dir = prep_logs(self.log_names)
        if self.log_names[1] == "fde":
            self.write_log()
        elif self.log_names[1] == "timing":
            self.write_timing()
        else:
            print("WARNING: no logs being written")

        return self.log_dir

    def calc_gt(self, data, gt):
        """Calculate ground truth fault status.

        Parameters
        ----------
        data : pd.DataFrame
            contains measurement information
        gt : pd.DataFrame
            ground truth position dataframe

        Returns
        -------
        data : pd.DataFrame
            contains measurement information now additionally with fault
            status
        mu_gt : np.array
            ground truth ECEF X,Y,Z and receiver clock bias estimate

        """
        epoch_gt = gt.iloc[abs(gt["millisSinceGpsEpoch"]
                 - data.loc[data.first_valid_index(),
                            "millisSinceGpsEpoch"]).argsort()[0]]
        gt_ecef = np.array([epoch_gt["ecefxM"],
                            epoch_gt["ecefyM"],
                            epoch_gt["ecefzM"]]).T

        # calculate "truth psuedorange"
        sat_pos = np.hstack((data["xSatPosM"].to_numpy().reshape(-1,1),
                             data["ySatPosM"].to_numpy().reshape(-1,1),
                             data["zSatPosM"].to_numpy().reshape(-1,1)))
        gt_pos = np.tile(gt_ecef,(sat_pos.shape[0],1))

        gt_psuedoranges = np.linalg.norm(gt_pos - sat_pos, axis = 1)

        # calculate receiver clock bias
        rb = 0.0
        rb = self.least_squares(gt_ecef,rb,data,True)

        # calculate residual
        data.loc[:,"residual"] = data["correctedPrM"].to_numpy() \
                               + data["satClkBiasM"].to_numpy() \
                               - gt_psuedoranges \
                               - rb

        # recalculate receiver clock bias
        residual_mean = data["residual"].mean()
        residual_std = data["residual"].std()
        distance_from_mean = abs(data["residual"] - residual_mean)
        data_rb = data[distance_from_mean <= residual_std]

        # calculate receiver clock bias
        rb = 0.0
        rb = self.least_squares(gt_ecef,rb,data_rb,True)

        data["rbM"] = rb

        # calculate residual
        data.loc[:,"residual"] = data["correctedPrM"].to_numpy() \
                               + data["satClkBiasM"].to_numpy() \
                               - gt_psuedoranges \
                               - rb


        data["fault"] = abs(data["residual"]) > FAULT_THRESHOLD

        mu_gt = np.concatenate((gt_ecef,np.array([rb])))

        return data, mu_gt

    def least_squares(self, x_est, rb, data, stationary = False,
                      for_ss = False):
        """Least squares estimation.

        Parameters
        ----------
        x_est:  np.array
            state estimate of ECEF X, Y, Z in shape [3 x 1]
        rb : float
            estimate of receiver clock bias
        data : pd.DataFrame
            DataFrame that contains information from the measurements
        stationary : bool
            If True, then only receiver clock bias is estimated/changed
        for_ss : bool
            If True, indicated that it LS is used for solution
            separation and will also return sigma^2 for the solution
            separation test statistic

        Returns
        -------
        rb : float
            receiver clock bias estimate
        x_est: np.array
            State estimate of ECEF X, Y, Z in shape [3 x 1]. Only
            Returned if stationary == False.
        sigma2 : float
            sigma^2 metric used in the test statistic of solution
            separation

        """
        count = 0
        delta = np.inf*np.ones((4,1))
        MAX_COUNT = 20

        numSats = len(data)

        sat_pos = np.hstack((data["xSatPosM"].to_numpy().reshape(-1,1),
                             data["ySatPosM"].to_numpy().reshape(-1,1),
                             data["zSatPosM"].to_numpy().reshape(-1,1)))

        x_est = x_est.copy()

        while np.linalg.norm(delta) > 1E-9:

            gt_pos = np.tile(x_est,(sat_pos.shape[0],1))

            gt_psuedoranges = np.linalg.norm(gt_pos - sat_pos, axis = 1)

            if stationary:
                G = np.ones((numSats,1))
            else:
                G = np.ones((numSats,4))
                G[:,:3] = np.divide(gt_pos - sat_pos,gt_psuedoranges.reshape(-1,1))

            # W = np.diag(1./data["rawPrUncM"]**2)
            W = np.diag(1./np.ones(len(data))**2)

            rho_diff = data["correctedPrM"].to_numpy() \
                     + data["satClkBiasM"].to_numpy() \
                     - gt_psuedoranges \
                     - rb

            rho_diff = rho_diff.reshape(-1,1)

            delta = np.linalg.pinv(W.dot(G)).dot(W).dot(rho_diff)

            if stationary:
                rb += delta[0,0]
            else:
                x_est += delta[0:3,0]
                rb += delta[3,0]

            count += 1
            # print(stationary,count,np.linalg.norm(delta))
            if count >= MAX_COUNT:
                break

        if stationary:
            return rb
        else:
            if for_ss:
                sigma2 = np.trace(np.linalg.inv(G.T.dot(G)))
                return x_est, rb, sigma2
            else:
                return x_est, rb

    def run_edm_fde(self, data, threshold, verbose=False):
        """EDM-FDE.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame that contains information from the measurements
        threshold : float
            detection threshold
        verbose : bool
            If True, prints extra debugging statements

        Returns
        -------
        data : pd.DataFrame
            Updated DataFrame with measurement faults indicated.

        """

        pranges = data["correctedPrM"].to_numpy() \
                + data["satClkBiasM"].to_numpy() \
                - self.mu["edm"][threshold][-1]
        pranges = pranges.reshape(1,-1)

        # calculate satellite positions
        S = np.vstack((data["xSatPosM"].to_numpy().reshape(1,-1),
                       data["ySatPosM"].to_numpy().reshape(1,-1),
                       data["zSatPosM"].to_numpy().reshape(1,-1)))

        D = edm_from_satellites_ranges(S,pranges)

        # EDM FDE
        dims = 3
        edm_faults = edm_fde(D, dims, 1, threshold, verbose)
        if verbose:
            print("edm faults: ",edm_faults)
        data.drop(edm_faults, inplace = True)

        return data

    def run_residual_fde(self, data, threshold, verbose = False):
        """Residual-based FDE.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame that contains information from the measurements
        threshold : float
            detection threshold
        verbose : bool
            If True, prints extra debugging statements

        Returns
        -------
        data : pd.DataFrame
            Updated DataFrame with measurement faults indicated.

        """
        sat_pos = np.hstack((data["xSatPosM"].to_numpy().reshape(-1,1),
                             data["ySatPosM"].to_numpy().reshape(-1,1),
                             data["zSatPosM"].to_numpy().reshape(-1,1)))
        mu_pos = np.tile(self.mu["residual"][threshold][:3].T,(sat_pos.shape[0],1))

        # calculate "truth psuedorange"
        gt_psuedoranges = np.linalg.norm(mu_pos - sat_pos, axis = 1)

        # calculate residual
        data.loc[:,"residual"] = data["correctedPrM"].to_numpy() \
                               + data["satClkBiasM"].to_numpy() \
                               - gt_psuedoranges \
                               - self.mu["residual"][threshold][-1]

        residuals = data.loc[:,"residual"].to_numpy().reshape(-1,1)

        # test statistic
        r = np.sqrt(residuals.T.dot(residuals)[0,0] \
                     / (len(data) - 4) )

        # iterate through subsets if r is above detection threshold
        if r > 10.:
            ri = set()
            r_subsets = []
            for ss in range(len(data)):
                residual_subset = np.delete(residuals,ss,axis=0)
                r_subset = np.sqrt(residual_subset.T.dot(residual_subset)[0,0] \
                             / (len(data) - 4) )
                if verbose:
                    r_subsets.append(r_subset)
                # adjusted threshold metric
                if r_subset/r < threshold:
                    ri.add(ss)

            if len(ri) >= 1:
                # residual_data = data[data.index not in list(ri)]
                residual_data = data.drop(list(ri))
            elif len(ri) == 0:
                if verbose:
                    print("r: ",r)
                    print("ri: ",ri)
                    for rri, rrr in enumerate(residuals):
                        print(rri, rrr, r_subsets[rri]/r)
                residual_data = data.copy()
        else:
            if verbose:
                print("r: ",r)
                for rri, rrr in enumerate(residuals):
                    print(rri, rrr)
            residual_data = data.copy()

        return residual_data

    def run_solution_fde(self, data, threshold, verbose=False,
                         recursion=False):
        """Residual-based FDE.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame that contains information from the measurements
        threshold : float
            detection threshold
        verbose : bool
            If True, prints extra debugging statements
        recursion : bool
            If True, then already in a recursive step and won't cause
            another recursion step to take place.

        Returns
        -------
        data : pd.DataFrame
            Updated DataFrame with measurement faults indicated.

        """
        solution_list = np.zeros((4,len(data)))

        wls_x = self.mu_history["solution"][threshold][:3,-1]
        wls_rb = self.mu_history["solution"][threshold][-1,-1]

        full_x, full_rb, sigma0 = self.least_squares(wls_x,
                                              wls_rb,
                                              data,False,True)

        full_solution = np.concatenate((full_x, np.array([full_rb]))).copy()


        normalizers = np.zeros((len(data),))

        for ss, sub in enumerate(list(data.index)):
            sub_data = data[data.index != sub]

            x_sub, rb_sub, sigmai = self.least_squares(wls_x,
                                            wls_rb,
                                            sub_data,False,True)

            sub_solution = np.concatenate((x_sub, np.array([rb_sub])))

            solution_list[:,ss] = sub_solution.copy()
            normalizers[ss] = sigmai

        full_matrix = np.tile(full_solution,(len(data),1)).T
        mean_diff = np.linalg.norm(solution_list - full_matrix,axis=0)
        normalizers = np.sqrt(normalizers - sigma0)
        test_statistic = np.divide(mean_diff,normalizers)



        if recursion:
            return np.sum(test_statistic > threshold) > 0

        if np.sum(test_statistic > threshold) > 0:
            worst_idx = np.argmax(test_statistic)
            sub_data = data[data.index != worst_idx]
            recurse_results = self.run_solution_fde(sub_data.copy(),threshold,
                                               verbose,True)
            if not recurse_results:
                solution_data = sub_data.copy()
            else:
                solution_data = data.copy()

        else:
            solution_data = data.copy()

        return solution_data

    def write_log(self):
        """Write statistics to log file.

        """

        for method in list(FDE_PARAMETERS.keys()):
            csv_filename = os.path.join(self.log_dir, self.trace_name + "-" \
                        + self.phone_type + "-" + method \
                        + "-ekf.csv")


            columns = ["parameter",
                       "tp", "tn", "fn", "fp",
                        ]

            with open(csv_filename, 'w') as csvfile:
                # creating a csv writer object
                csvwriter = csv.writer(csvfile)

                # writing the fields
                csvwriter.writerow(columns)
                for param in FDE_PARAMETERS[method]:
                    data = [param,
                            self.fde_accuracy[method][param]["tp"],
                            self.fde_accuracy[method][param]["tn"],
                            self.fde_accuracy[method][param]["fn"],
                            self.fde_accuracy[method][param]["fp"],
                            ]
                    csvwriter.writerow(data)

    def write_timing(self):
        """Write timing to log file.

        """

        for method in list(FDE_PARAMETERS.keys()):
            csv_filename = os.path.join(self.log_dir, self.trace_name + "-" \
                        + self.phone_type + "-" + method \
                        + "-timing.csv")


            columns = self.timing_history[method].keys()

            with open(csv_filename, 'w') as csvfile:
                # creating a csv writer object
                csvwriter = csv.writer(csvfile)

                # writing the fields
                csvwriter.writerow(columns)
                for value in self.timing_history[method].values():
                    csvwriter.writerow(value)
