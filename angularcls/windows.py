from typing import Union, Callable

from scipy import interpolate as sinterp

import numpy as np

from . import cosmoconstants


def interpolate(zs: np.ndarray, window: np.ndarray, interp1d: bool):
    if interp1d:
        window = sinterp.interp1d(zs, window)
    return window

def cmblensingwindow_ofchi(chis: np.ndarray, aofchis: np.ndarray, H0: float, Omegam: float, interp1d: bool = True, chistar:float = cosmoconstants.CHISTAR, cSpeedKmPerSec: float = cosmoconstants.CSPEEDKMPERSEC) -> Union[np.ndarray, Callable]:
    window = 1.5*(Omegam)*H0**2*chis*((chistar - chis)/chistar)/cSpeedKmPerSec**2/aofchis
    return interpolate(chis, window, interp1d)

def cmblensingwindow(zs: np.ndarray, chis: np.ndarray, Hzs: np.ndarray, H0: float, Omegam: float, interp1d: bool = True, chistar:float = cosmoconstants.CHISTAR, cSpeedKmPerSec: float = cosmoconstants.CSPEEDKMPERSEC) -> Union[np.ndarray, Callable]:
    window = 1.5*(Omegam)*H0**2*(1.+zs)*chis*((chistar - chis)/chistar)/Hzs/cSpeedKmPerSec
    return interpolate(zs, window, interp1d)


def gaussianwindow(zs: np.ndarray, mean: float, sigma: float, interp1d: bool = True):
    window = np.exp(-(zs - mean)**2 / 2 / sigma**2) / np.sqrt(2*np.pi)/sigma
    return interpolate(zs, window, interp1d)


def magnificationbiaswindow(s: float, wg: Callable, Hzs: np.ndarray, H0: float, Omegam0: float, zs: np.ndarray, chis: np.ndarray, cSpeedKmPerSec: float = cosmoconstants.CSPEEDKMPERSEC):
    magwindow = lensingwindow(wg, zs, chis)
    bias = (5*s-2)
    factor = 3/2/cSpeedKmPerSec*(Omegam0)*H0**2*(1.+zs)/Hzs
    return bias*factor*magwindow

def lensingwindow(wg: Callable, zs: np.ndarray, chis: np.ndarray): 
    '''
    Assumes last of zs is around zstar.
    '''
    chiofz = sinterp.interp1d(zs, chis)
    def obtain_single_element(i, z):
        integrand = lambda zprime: (chiofz(zprime)-chiofz(z))/chiofz(zprime)*wg(zprime)
        z_ = zs[i:]
        result = np.trapz(integrand(z_), z_)
        return np.nan_to_num(result)*chiofz(z)

    return [obtain_single_element(i, z) for i, z in enumerate(zs)]

