########################################################################
# Author(s):    Shubh Gupta, Ashwin Kanhere
# Date:         20 July 2021
# Desc:         Functions for coordinate conversions required by GPS
########################################################################

import os
import sys
# append <path>/gnss_lib_py/gnss_lib_py/ to path
sys.path.append(os.path.dirname(
                os.path.dirname(
                os.path.realpath(__file__))))
import numpy as np

#Coordinate conversions (From https://github.com/commaai/laika)

class CoordConsts:
    """Class containing constants required for coordinate conversion.

    Attributes
    ----------
    A : float
        Semi-major axis of the earth [m]
    B : float
        Semi-minor axis of the earth [m]
    ESQ : float
        First esscentricity squared
    ESQ1: float
        Second eccentricity squared
    """
    # TODO: Update docstring for ESQ and ESQ1

    def __init__(self):
        self.A = 6378137.
        self.B = 6356752.3145
        self.ESQ = 6.69437999014 * 0.001
        self.E1SQ = 6.73949674228 * 0.001

def geodetic2ecef(geodetic, radians=False):
    """LLA to ECEF conversion

    Parameters
    ----------
    geodetic : ndarray
        Float with WGS-84 LLA coordinates
    radians : bool
        Flag of whether input is in radians

    Returns
    -------
    ecef : ndarray
        ECEF coordinates corresponding to input LLA

    Notes
    -----
    Based on code from https://github.com/commaai/laika

    """
    coordconsts = CoordConsts()
    geodetic = np.array(geodetic)
    input_shape = geodetic.shape
    geodetic = np.atleast_2d(geodetic)

    ratio = 1.0 if radians else (np.pi / 180.0)
    lat = ratio*geodetic[:,0]
    lon = ratio*geodetic[:,1]
    alt = geodetic[:,2]

    xi = np.sqrt(1 - coordconsts.ESQ * np.sin(lat)**2)
    x = (coordconsts.A / xi + alt) * np.cos(lat) * np.cos(lon)
    y = (coordconsts.A / xi + alt) * np.cos(lat) * np.sin(lon)
    z = (coordconsts.A / xi * (1 - coordconsts.ESQ) + alt) * np.sin(lat)
    ecef = np.array([x, y, z]).T
    ecef = np.reshape(ecef, input_shape)
    return ecef


def ecef2geodetic(ecef, radians=False):
    """ECEF to LLA conversion using Ferrari's method

    Parameters
    ----------
    ecef : ndarray
        Float with ECEF coordinates
    radians : bool
        Flag of whether output should be in radians

    Returns
    -------
    geodetic : ndarray
        Float with WGS-84 LLA coordinates corresponding to input ECEF

    Notes
    -----
    Based on code from https://github.com/commaai/laika

    """
    coordconsts = CoordConsts()
    ecef = np.atleast_1d(ecef)
    input_shape = ecef.shape
    ecef = np.atleast_2d(ecef)
    x, y, z = ecef[:, 0], ecef[:, 1], ecef[:, 2]

    ratio = 1.0 if radians else (180.0 / np.pi)

    # Convert from ECEF to geodetic using Ferrari's methods
    # https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#Ferrari.27s_solution
    r = np.sqrt(x * x + y * y)
    Esq = coordconsts.A * coordconsts.A - coordconsts.B * coordconsts.B
    F = 54 * coordconsts.B * coordconsts.B * z * z
    G = r * r + (1 - coordconsts.ESQ) * z * z - coordconsts.ESQ * Esq
    C = (coordconsts.ESQ * coordconsts.ESQ * F * r * r) / (pow(G, 3))
    S = np.cbrt(1 + C + np.sqrt(C * C + 2 * C))
    P = F / (3 * pow((S + 1 / S + 1), 2) * G * G)
    Q = np.sqrt(1 + 2 * coordconsts.ESQ * coordconsts.ESQ * P)
    r_0 =  -(P * coordconsts.ESQ * r) / (1 + Q) + np.sqrt(0.5 * coordconsts.A * coordconsts.A*(1 + 1.0 / Q) - \
          P * (1 - coordconsts.ESQ) * z * z / (Q * (1 + Q)) - 0.5 * P * r * r)
    U = np.sqrt(pow((r - coordconsts.ESQ * r_0), 2) + z * z)
    V = np.sqrt(pow((r - coordconsts.ESQ * r_0), 2) + (1 - coordconsts.ESQ) * z * z)
    Z_0 = coordconsts.B * coordconsts.B * z / (coordconsts.A * V)
    h = U * (1 - coordconsts.B * coordconsts.B / (coordconsts.A * V))
    lat = ratio*np.arctan((z + coordconsts.E1SQ * Z_0) / r)
    lon = ratio*np.arctan2(y, x)

    # stack the new columns and return to the original shape
    geodetic = np.column_stack((lat, lon, h))
    geodetic = np.reshape(geodetic, input_shape)
    return geodetic.reshape(input_shape)
