from typing import Union, Callable

from scipy import interpolate as sinterp
import jax_cosmo.scipy.interpolate as jsinterp

import jax

import numpy as np

from astropy import units as u
from astropy.cosmology import Planck15

from . import cosmoconstants


def interpolate(zs: np.ndarray, window: np.ndarray, interp1d: bool):
    if interp1d:
        window = sinterp.interp1d(zs, window)
    else:
        window = jax.jit(jsinterp.InterpolatedUnivariateSpline(zs, window, k = 3))
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



def cmblensingwindow_ofchi(chis: np.ndarray, aofchis: np.ndarray, H0: float, Omegam: float, interp1d: bool = True, chistar:float = cosmoconstants.CHISTAR, cSpeedKmPerSec: float = cosmoconstants.CSPEEDKMPERSEC) -> Union[np.ndarray, Callable]:
    window = 1.5*(Omegam)*H0**2*chis*((chistar - chis)/chistar)/cSpeedKmPerSec**2/aofchis
    return interpolate(chis, window, interp1d)



def cmblensingwindow_ofchi_truncated(chis: np.ndarray, aofchis: np.ndarray, H0: float, Omegam: float, interp1d: bool = True, chistar:float = cosmoconstants.CHISTAR, cSpeedKmPerSec: float = cosmoconstants.CSPEEDKMPERSEC, zs = None, zthreshold = cosmoconstants.CHISTAR) -> Union[np.ndarray, Callable]:
    window = 1.5*(Omegam)*H0**2*chis*((chistar - chis)/chistar)/cSpeedKmPerSec**2/aofchis*(zs <= zthreshold)
    return interpolate(chis, window, interp1d)


@np.vectorize(excluded = ['beta', 'alpha'])
def f_nu(f, beta, alpha = 2):
    T = 34
    k = 1.38064852e-23
    h = 6.62607015e-34

    nu0 = 4955*10**9
    
    if f < nu0:
        result = (np.exp(h*f/(k*T))-1)**(-1)*f**(beta+3)
    elif f >= nu0:
        result = (np.exp(h*nu0/(k*T))-1)**(-1)*f**(beta+3)*(f/(nu0))**(-alpha)

    return result


def cibwindow_ofchi(chis, aofchis, H0, Omegam, interp1d, chistar, zs, nu, b_c = 3.6e-62):
    
    z_c = 2
    sigma_z = 2
    beta = 2
    nu = nu*10**9

    freq = nu * u.Hz
    equiv = u.thermodynamic_temperature(freq, Planck15.Tcmb0)
    conv = (1. * u.Jy/ u.sr).to(u.uK, equivalencies = equiv)  #convert from J/sr to uk

    window = b_c*(chis**2/(1.+zs)**2)*np.exp(-(zs-z_c)**2/(2*sigma_z**2))*f_nu(nu*(1+zs), beta)*conv.value

    return interpolate(chis, window, interp1d)