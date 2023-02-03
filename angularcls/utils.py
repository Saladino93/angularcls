import numpy as np

import camb
from camb import model as cmodel

def growth_factor_quick(zs, pars, kmax):
    zs_ = np.logspace(-9, np.log(1089), 150)
    pars2 = pars.copy() 
    pars2.set_matter_power(redshifts = zs_, kmax = kmax)

    #Linear spectra
    pars2.NonLinear = cmodel.NonLinear_none
    results2 = camb.get_results(pars2)
    results2.get_matter_power_spectrum(minkh = 1e-4, maxkh = 1, npoints = 200)
    s8 = np.array(results2.get_sigma8())
    growth = s8[::-1]/s8[::-1][0]
    return np.interp(zs, zs_, growth)