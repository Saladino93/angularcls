from typing import Union, Callable

from scipy import interpolate as sinterp

import numpy as np

from . import cosmoconstants


def interpolate(zs: np.ndarray, window: np.ndarray, interp1d: bool):
    if interp1d:
        window = sinterp.interp1d(zs, window)
    return window

def cmblensingwindow(zs: np.ndarray, chis: np.ndarray, Hzs: np.ndarray, H0: float, Omegam: float, interp1d: bool = True, chistar:float = cosmoconstants.CHISTAR, cSpeedKmPerSec: float = cosmoconstants.CSPEEDKMPERSEC) -> Union[np.ndarray, Callable]:
    window = 1.5*(Omegam)*H0**2*(1.+zs)*chis*((chistar - chis)/chistar)/Hzs/cSpeedKmPerSec
    return interpolate(zs, window, interp1d)


def gaussianwindow(zs: np.ndarray, mean: float, sigma: float, interp1d: bool = True):
    window = np.exp(-(zs - mean)**2 / 2 / sigma**2) / np.sqrt(2*np.pi)/sigma
    return interpolate(zs, window, interp1d)