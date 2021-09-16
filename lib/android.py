########################################################################
# Author(s):    Shubh Gupta, Adam Dai
# Date:         13 Jul 2021
# Desc:         Functions to process Android measurements
########################################################################

import os
import sys
# append <path>/gnss_lib_py/gnss_lib_py/ to path
sys.path.append(os.path.dirname(
                os.path.dirname(
                os.path.realpath(__file__))))
import csv
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class GPSConsts:
    """Class containing constants required for GPS navigation.

    Based on ECE 456 implementation [1]_.

    Attributes
    ----------
    A : float
        Semi-major axis of the earth [m]
    B : float
        Semi-minor axis of the earth [m]
    E : float
        Eccentricity of the earth = 0.08181919035596
    LAT_ACC_THRESH : float
        10 meter latitude accuracy
    MUEARTH : float
        :math:`G*M_E`, the "gravitational constant" for orbital
        motion about the Earth [m^3/s^2]
    OMEGAEDOT : float
        The sidereal rotation rate of the Earth
        (WGS-84) [rad/s]
    C : float
        speed of light [m/s]
    F : float
        Relativistic correction term [s/m^(1/2)]
    F1 : float
        GPS L1 frequency [Hz]
    F2 : float
        GPS L2 frequency [Hz]
    PI : float
        pi
    T_TRANS : float
        70 ms is the average time taken for signal transmission from GPS sats
    GRAV : float
        Acceleration due to gravity ENU frame of reference [m/s]
    WEEKSEC : float
        Number of seconds in a week [s]

    References
    ----------
    .. [1] Makela, Jonathan, ECE 456, Global Nav Satellite Systems, Fall 2017. 
      University of Illinois Urbana-Champaign. Coding Assignments.

    """
    def __init__(self):
        self.A = 6378137.
        self.B = 6356752.3145
        self.E = np.sqrt(1-(self.B**2)/(self.A**2))
        self.LAT_ACC_THRESH = 1.57e-6
        self.MUEARTH = 398600.5e9
        self.OMEGAEDOT = 7.2921151467e-5
        self.C = 299792458.
        self.F = -4.442807633e-10
        self.F1 = 1.57542e9
        self.F2 = 1.22760e9
        self.PI = 3.1415926535898
        self.T_TRANS = 70*0.001
        self.GRAV = -9.80665
        self.WEEKSEC = 604800

def make_gnss_dataframe(input_path, verbose=False):
    """Read Android raw file and produce gnss dataframe objects

    Parameters
    ----------
    input_path : string
        File location of data file to read.
    verbose : bool
        If true, will print out any problems that were detected.

    Returns
    -------
    corrected_measurements : pandas dataframe
        Dataframe that contains a corrected version of the measurements.
    andorid fixes : pandas dataframe
        Dataframe that contains the andorid fixes from the log file.

    """
    with open(input_path) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[0][0] == '#':
                if 'Fix' in row[0]:
                    android_fixes = [row[1:]]
                elif 'Raw' in row[0]:
                    measurements = [row[1:]]
            else:
                if row[0] == 'Fix':
                    android_fixes.append(row[1:])
                elif row[0] == 'Raw':
                    measurements.append(row[1:])

    android_fixes = pd.DataFrame(android_fixes[1:], columns = android_fixes[0])
    measurements = pd.DataFrame(measurements[1:], columns = measurements[0])

    corrected_measurements = correct_log(measurements, verbose=verbose)

    return corrected_measurements, android_fixes

def correct_log(measurements, verbose=False, carrier_phase_checks = False):
    """Compute required quantities from the log and check for errors.

    This is a master function that calls the other correction functions
    to validate a GNSS measurement log dataframe.

    Parameters
    ----------
    measurements : pandas dataframe
        pandas dataframe that holds gnss meassurements
    verbose : bool
        If true, will print out any problems that were detected.
    carrier_phase_checks : bool
        If true, completes carrier phase checks

    Returns
    -------
    measurements : pandas dataframe
        same dataframe with possibly some fixes or column additions

    """
    # Add leading 0
    measurements.loc[measurements['Svid'].str.len() == 1, 'Svid'] = '0' + measurements['Svid']

    # Compatibility with RINEX files
    measurements.loc[measurements['ConstellationType'] == '1', 'Constellation'] = 'G'
    measurements.loc[measurements['ConstellationType'] == '3', 'Constellation'] = 'R'
    measurements['SvName'] = measurements['Constellation'] + measurements['Svid']

    # Drop non-GPS measurements
    measurements = measurements.loc[measurements['Constellation'] == 'G']

    # TODO: Measurements should be discarded if the constellation is unknown.

    # Convert columns to numeric representation
    # assign() function prevents SettingWithCopyWarning
    measurements = measurements.assign(Cn0DbHz=pd.to_numeric(measurements['Cn0DbHz']))
    measurements = measurements.assign(TimeNanos=pd.to_numeric(measurements['TimeNanos']))
    measurements = measurements.assign(FullBiasNanos=pd.to_numeric(measurements['FullBiasNanos']))
    measurements = measurements.assign(ReceivedSvTimeNanos=pd.to_numeric(measurements['ReceivedSvTimeNanos']))
    measurements = measurements.assign(PseudorangeRateMetersPerSecond=pd.to_numeric(measurements['PseudorangeRateMetersPerSecond']))
    measurements = measurements.assign(ReceivedSvTimeUncertaintyNanos=pd.to_numeric(measurements['ReceivedSvTimeUncertaintyNanos']))

    # Check clock fields
    error_logs = []
    measurements, error_logs = check_gnss_clock(measurements, error_logs)
    measurements, error_logs = check_gnss_measurements(measurements, error_logs)
    if carrier_phase_checks:
        measurements, error_logs = check_carrier_phase(measurements, error_logs)
    measurements, error_logs = compute_times(measurements, error_logs)
    measurements, error_logs = compute_pseudorange(measurements, error_logs)

    if verbose:
        if len(error_logs)>0:
            print("Following problems detected:")
            print(error_logs)
        else:
            print("No problems detected.")
    return measurements

def check_gnss_clock(gnssRaw, gnssAnalysis):
    """Checks and fixes clock field errors

    Additonal checks added from [1]_.

    Parameters
    ----------
    gnssRaw : pandas dataframe
        pandas dataframe that holds gnss meassurements
    gnssAnalysis : list
        holds any error messages

    Returns
    -------
    gnssRaw : pandas dataframe
        same dataframe with possibly some fixes or column additions
    gnssAnalysis : list
        holds any error messages

    Notes
    -----
    Based off of Matlab code from Google's gps-measurement-tools
    repository: https://github.com/google/gps-measurement-tools. Compare
    with CheckGnssClock() in opensource/ReadGnssLogger.m

    References
    ----------
    .. [1] Fu, Guoyu Michael, Mohammed Khider, and Frank van Diggelen.
       "Android Raw GNSS Measurement Datasets for Precise Positioning."
       Proceedings of the 33rd International Technical Meeting of the
       Satellite Division of The Institute of Navigation (ION GNSS+
       2020). 2020.

    """
    # list of clock fields
    gnssClockFields = [
      'TimeNanos',
      'TimeUncertaintyNanos',
      'TimeOffsetNanos',
      'LeapSecond',
      'FullBiasNanos',
      'BiasUncertaintyNanos',
      'DriftNanosPerSecond',
      'DriftUncertaintyNanosPerSecond',
      'HardwareClockDiscontinuityCount',
      'BiasNanos'
      ]
    for field in gnssClockFields:
        if field not in gnssRaw.head():
            gnssAnalysis.append('WARNING: '+field+' (Clock) is missing from GNSS Logger file')
        else:
            gnssRaw.loc[:,field] = pd.to_numeric(gnssRaw[field])
    ok = all(x in gnssRaw.head() for x in ['TimeNanos', 'FullBiasNanos'])
    if not ok:
        gnssAnalysis.append('FAIL Clock check')
        return gnssRaw, gnssAnalysis

    # Measurements should be discarded if TimeNanos is empty
    if gnssRaw["TimeNanos"].isnull().values.any():
        gnssRaw.dropna(how = "any", subset = ["TimeNanos"],
                       inplace = True)
        gnssAnalysis.append('empty or invalid TimeNanos')

    if 'BiasNanos' not in gnssRaw.head():
        gnssRaw.loc[:,'BiasNanos'] = 0
    if 'TimeOffsetNanos' not in gnssRaw.head():
        gnssRaw.loc[:,'TimeOffsetNanos'] = 0
    if 'HardwareClockDiscontinuityCount' not in gnssRaw.head():
        gnssRaw.loc[:,'HardwareClockDiscontinuityCount'] = 0
        gnssAnalysis.append('WARNING: Added HardwareClockDiscontinuityCount=0 because it is missing from GNSS Logger file')

    # measurements should be discarded if FullBiasNanos is zero or invalid
    if any(gnssRaw.FullBiasNanos >= 0):
        gnssRaw.FullBiasNanos = -1*gnssRaw.FullBiasNanos
        gnssAnalysis.append('WARNING: FullBiasNanos wrong sign. Should be negative. Auto changing inside check_gnss_clock')

    # Measurements should be discarded if BiasUncertaintyNanos is too
    # large ## TODO: figure out how to choose this parameter better
    if any(gnssRaw.BiasUncertaintyNanos >= 40.):
        count = (gnssRaw["BiasUncertaintyNanos"] >= 40.).sum()
        gnssAnalysis.append(str(count) +
         ' rows with too large BiasUncertaintyNanos')
        gnssRaw = gnssRaw[gnssRaw["BiasUncertaintyNanos"] < 40.]


    gnssRaw = gnssRaw.assign(allRxMillis = ((gnssRaw.TimeNanos - gnssRaw.FullBiasNanos)/1e6))
    # gnssRaw['allRxMillis'] = ((gnssRaw.TimeNanos - gnssRaw.FullBiasNanos)/1e6)
    return gnssRaw, gnssAnalysis


def check_gnss_measurements(gnssRaw, gnssAnalysis):
    """Checks that GNSS measurement fields exist in dataframe.

    Additonal checks added from [1]_.

    Parameters
    ----------
    gnssRaw : pandas dataframe
        pandas dataframe that holds gnss meassurements
    gnssAnalysis : list
        holds any error messages

    Returns
    -------
    gnssRaw : pandas dataframe
        exact same dataframe as input (why is this a return?)
    gnssAnalysis : list
        holds any error messages

    Notes
    -----
    Based off of Matlab code from Google's gps-measurement-tools
    repository: https://github.com/google/gps-measurement-tools. Compare
    with ReportMissingFields() in opensource/ReadGnssLogger.m

    References
    ----------
    .. [1] Fu, Guoyu Michael, Mohammed Khider, and Frank van Diggelen.
       "Android Raw GNSS Measurement Datasets for Precise Positioning."
       Proceedings of the 33rd International Technical Meeting of the
       Satellite Division of The Institute of Navigation (ION GNSS+
       2020). 2020.

    """
    # list of measurement fields
    gnssMeasurementFields = [
        'Cn0DbHz',
        'ConstellationType',
        'MultipathIndicator',
        'PseudorangeRateMetersPerSecond',
        'PseudorangeRateUncertaintyMetersPerSecond',
        'ReceivedSvTimeNanos',
        'ReceivedSvTimeUncertaintyNanos',
        'State',
        'Svid',
        'AccumulatedDeltaRangeMeters',
        'AccumulatedDeltaRangeUncertaintyMeters'
        ]
    for field in gnssMeasurementFields:
        if field not in gnssRaw.head():
            gnssAnalysis.append('WARNING: '+field+' (Measurement) is missing from GNSS Logger file')

    # measurements should be discarded if state is neither
    # STATE_TOW_DECODED nor STATE_TOW_KNOWN
    gnssRaw = gnssRaw.assign(State=pd.to_numeric(gnssRaw['State']))
    STATE_TOW_DECODED = 0x8
    STATE_TOW_KNOWN = 0x4000
    invalid_state_count = np.invert((gnssRaw["State"] & STATE_TOW_DECODED).astype(bool) |
                              (gnssRaw["State"] & STATE_TOW_KNOWN).astype(bool)).sum()
    if invalid_state_count > 0:
        gnssAnalysis.append(str(invalid_state_count) + " rows have " + \
                            "state TOW neither decoded nor known")
        gnssRaw = gnssRaw[(gnssRaw["State"] & STATE_TOW_DECODED).astype(bool) |
                          (gnssRaw["State"] & STATE_TOW_KNOWN).astype(bool)]

    # Measurements should be discarded if ReceivedSvTimeUncertaintyNanos
    # is high ## TODO: figure out how to choose this parameter better
    if any(gnssRaw.ReceivedSvTimeUncertaintyNanos >= 150.):
        count = (gnssRaw["ReceivedSvTimeUncertaintyNanos"] >= 150.).sum()
        gnssAnalysis.append(str(count) +
         ' rows with too large ReceivedSvTimeUncertaintyNanos')
        gnssRaw = gnssRaw[gnssRaw["ReceivedSvTimeUncertaintyNanos"] < 150.]

    # convert multipath indicator to numeric
    gnssRaw = gnssRaw.assign(MultipathIndicator=pd.to_numeric(gnssRaw['MultipathIndicator']))

    return gnssRaw, gnssAnalysis

def check_carrier_phase(gnssRaw, gnssAnalysis):
    """Checks that carrier phase measurements.

    Checks taken from [1]_.

    Parameters
    ----------
    gnssRaw : pandas dataframe
        pandas dataframe that holds gnss meassurements
    gnssAnalysis : list
        holds any error messages

    Returns
    -------
    gnssRaw : pandas dataframe
        exact same dataframe as input (why is this a return?)
    gnssAnalysis : list
        holds any error messages

    References
    ----------
    .. [1] Fu, Guoyu Michael, Mohammed Khider, and Frank van Diggelen.
       "Android Raw GNSS Measurement Datasets for Precise Positioning."
       Proceedings of the 33rd International Technical Meeting of the
       Satellite Division of The Institute of Navigation (ION GNSS+
       2020). 2020.

    """

    # Measurements should be discarded if AdrState violates
    # ADR_STATE_VALID == 1 & ADR_STATE_RESET == 0
    # & ADR_STATE_CYCLE_SLIP == 0
    gnssRaw = gnssRaw.assign(AccumulatedDeltaRangeState=pd.to_numeric(gnssRaw['AccumulatedDeltaRangeState']))
    ADR_STATE_VALID = 0x1
    ADR_STATE_RESET = 0x2
    ADR_STATE_CYCLE_SLIP = 0x4

    invalid_state_count = np.invert((gnssRaw["AccumulatedDeltaRangeState"] & ADR_STATE_VALID).astype(bool) &
                          np.invert((gnssRaw["AccumulatedDeltaRangeState"] & ADR_STATE_RESET).astype(bool)) &
                          np.invert((gnssRaw["AccumulatedDeltaRangeState"] & ADR_STATE_CYCLE_SLIP).astype(bool))).sum()
    if invalid_state_count > 0:
        gnssAnalysis.append(str(invalid_state_count) + " rows have " + \
                            "ADRstate invalid")
        gnssRaw = gnssRaw[(gnssRaw["AccumulatedDeltaRangeState"] & ADR_STATE_VALID).astype(bool) &
                          np.invert((gnssRaw["AccumulatedDeltaRangeState"] & ADR_STATE_RESET).astype(bool)) &
                          np.invert((gnssRaw["AccumulatedDeltaRangeState"] & ADR_STATE_CYCLE_SLIP).astype(bool))]

    # Measurements should be discarded if AccumulatedDeltaRangeUncertaintyMeters
    # is too large ## TODO: figure out how to choose this parameter better
    gnssRaw = gnssRaw.assign(AccumulatedDeltaRangeUncertaintyMeters=pd.to_numeric(gnssRaw['AccumulatedDeltaRangeUncertaintyMeters']))
    if any(gnssRaw.AccumulatedDeltaRangeUncertaintyMeters >= 0.15):
        count = (gnssRaw["AccumulatedDeltaRangeUncertaintyMeters"] >= 0.15).sum()
        gnssAnalysis.append(str(count) +
         ' rows with too large AccumulatedDeltaRangeUncertaintyMeters')
        gnssRaw = gnssRaw[gnssRaw["AccumulatedDeltaRangeUncertaintyMeters"] < 0.15]

    return gnssRaw, gnssAnalysis


def compute_times(gnssRaw, gnssAnalysis):
    """Compute times and epochs for GNSS measurements.

    Additional checks added from [1]_.

    Parameters
    ----------
    gnssRaw : pandas dataframe
        pandas dataframe that holds gnss meassurements
    gnssAnalysis : list
        holds any error messages

    Returns
    -------
    gnssRaw : pandas dataframe
        Dataframe with added columns updated.
    gnssAnalysis : list
        Holds any error messages. This function doesn't actually add any
        error messages, but it is a nice thought.

    Notes
    -----
    Based off of Matlab code from Google's gps-measurement-tools
    repository: https://github.com/google/gps-measurement-tools. Compare
    with opensource/ProcessGnssMeas.m

    References
    ----------
    .. [1] Fu, Guoyu Michael, Mohammed Khider, and Frank van Diggelen.
       "Android Raw GNSS Measurement Datasets for Precise Positioning."
       Proceedings of the 33rd International Technical Meeting of the
       Satellite Division of The Institute of Navigation (ION GNSS+
       2020). 2020.

    """
    gpsepoch = datetime(1980, 1, 6, 0, 0, 0)
    WEEKSEC = 604800
    gnssRaw['GpsWeekNumber'] = np.floor(-1*gnssRaw['FullBiasNanos']*1e-9/WEEKSEC)
    gnssRaw['GpsTimeNanos'] = gnssRaw['TimeNanos'] - (gnssRaw['FullBiasNanos'] + gnssRaw['BiasNanos'])

    # Measurements should be discarded if arrival time is negative
    if sum(gnssRaw['GpsTimeNanos'] <= 0) > 0:
        gnssRaw = gnssRaw[gnssRaw['GpsTimeNanos'] > 0]
        gnssAnalysis.append("negative arrival times removed")
    # TODO: Measurements should be discarded if arrival time is
    # unrealistically large

    gnssRaw['tRxNanos'] = (gnssRaw['TimeNanos']+gnssRaw['TimeOffsetNanos'])-(gnssRaw['FullBiasNanos'].iloc[0]+gnssRaw['BiasNanos'].iloc[0])
    gnssRaw['tRxSeconds'] = 1e-9*gnssRaw['tRxNanos'] - WEEKSEC * gnssRaw['GpsWeekNumber']
    gnssRaw['tTxSeconds'] = 1e-9*(gnssRaw['ReceivedSvTimeNanos'] + gnssRaw['TimeOffsetNanos'])
    gnssRaw['LeapSecond'] = gnssRaw['LeapSecond'].fillna(0)
    gnssRaw["UtcTimeNanos"] = pd.to_datetime(gnssRaw['GpsTimeNanos']  - gnssRaw["LeapSecond"] * 1E9, utc = True, origin=gpsepoch)
    gnssRaw['UnixTime'] = pd.to_datetime(gnssRaw['GpsTimeNanos'], utc = True, origin=gpsepoch)
    # TODO: Check if UnixTime is the same as UtcTime, if so remove it

    gnssRaw['Epoch'] = 0
    gnssRaw.loc[gnssRaw['UnixTime'] - gnssRaw['UnixTime'].shift() > timedelta(milliseconds=200), 'Epoch'] = 1
    gnssRaw['Epoch'] = gnssRaw['Epoch'].cumsum()
    return gnssRaw, gnssAnalysis

def compute_pseudorange(gnssRaw, gnssAnalysis):
    """Compute psuedorange values and add to dataframe.

    Parameters
    ----------
    gnssRaw : pandas dataframe
        pandas dataframe that holds gnss meassurements
    gnssAnalysis : list
        holds any error messages

    Returns
    -------
    gnssRaw : pandas dataframe
        Dataframe with added columns updated.
    gnssAnalysis : list
        Holds any error messages. This function doesn't actually add any
        error messages, but it is a nice thought.

    Notes
    -----
    Based off of Matlab code from Google's gps-measurement-tools
    repository: https://github.com/google/gps-measurement-tools. Compare
    with opensource/ProcessGnssMeas.m

    """

    gpsconsts = GPSConsts()
    gnssRaw['Pseudorange_seconds'] = gnssRaw['tRxSeconds'] - gnssRaw['tTxSeconds']
    gnssRaw['Pseudorange_meters'] = gnssRaw['Pseudorange_seconds']*gpsconsts.C
    gnssRaw['Pseudorange_sigma_meters'] = gpsconsts.C * 1e-9 * gnssRaw['ReceivedSvTimeUncertaintyNanos']
    return gnssRaw, gnssAnalysis
