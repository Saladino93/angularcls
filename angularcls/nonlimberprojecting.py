import numpy as np

import scipy.integrate as sintegrate

import scipy.interpolate as sinterp

import sys
sys.path.append('/Users/omard/Documents/projects/FFTLog-and-beyond/python/.')
import fftlog

from . import cosmoconstants, limberprojecting, projector

from typing import Callable, Union, List, Set



class FFTlogProjector(limberprojecting.LimberProjector):
    """
    Based on equation (2.10) of https://arxiv.org/pdf/1911.11947.pdf

    Below equation (4.5) some info for the FFTlog method parameters is given.
    """

    def __init__(self, nu: float = 1.01, N_extrap_low: int = 0, N_extrap_high:int = 0, c_window_width: float = 0.25, N_pad: int = 1000, **kwargs):

        super().__init__(**kwargs)
        self.nu = nu
        self.N_extrap_low = N_extrap_low
        self.N_extrap_high = N_extrap_high
        self.c_window_width = c_window_width
        self.N_pad = N_pad


    def get_log_object(self, chilog, flog):
        myfftlog = fftlog.fftlog(chilog, flog, nu = self.nu, N_extrap_low = self.N_extrap_low, N_extrap_high = self.N_extrap_high, c_window_width = self.c_window_width, N_pad = self.N_pad)
        return myfftlog

    def inputs_from_z_functions(self, zs, chis, hubble, W, growth, bias):
        """
        Parameters
        ----------
        zs : array_like
            Array of redshift values.
        chis : array_like
            Array of comoving distance values.
        hubble : function
            Function of redshift returning the Hubble parameter.
        W : function
            Function of redshift returning the window function.
        growth : function
            Function of redshift returning the growth function.
        """
        quickinterp = lambda function: sinterp.interp1d(chis, function(zs), fill_value = 0., bounds_error = False)
        return list(map(quickinterp, [hubble, W, growth, bias]))

    def get_log_object_from_inputs(self, chilog, bofchi, Hofchi, nz0ofchi, growthofchi):
        f = lambda chi: chi*bofchi(chi)*Hofchi(chi)*nz0ofchi(chi)*growthofchi(chi)/cosmoconstants.CSPEEDKMPERSEC
        return self.get_log_object(chilog, f(chilog))

    def get_log_object_from_inputs_from_z_functions(self, chilog, zs, chis, hubble, W, growth, bias):
        return self.get_log_object_from_inputs(chilog, *self.inputs_from_z_functions(zs, chis, hubble, W, growth, bias))

    def integrate(self, ls: np.ndarray, zs: np.ndarray, chis: np.ndarray, hubble: Callable, growth: Callable, bias_A: Callable, bias_B: Callable, window_A: Callable, window_B: Callable, linear_power_interpolator: Callable, non_linear_power_interpolator: Callable, 
                  chi_min: float, chi_max: float, num: int = 1000, ls_limber: int = 100, miniter: int = 1500):
        """
        Parameters
        ----------
        ls : array_like
            Array of multipole values. 
        zs : array_like
            Array of redshift values.
        chis : array_like
            Array of comoving distance values.
        hubble : Callable
            Function of redshift returning the Hubble parameter.
        growth : Callable
            Function of redshift returning the growth function.
        bias_A : Callable
            Function of redshift returning the bias of the first field.
        bias_B : Callable
            Function of redshift returning the bias of the second field.
        window_A : Callable
            Function of redshift returning the window function of the first field.
        window_B : Callable
            Function of redshift returning the window function of the second field.
        linear_power_interpolator : Callable
            Function of k and z returning the linear power spectrum.
        non_linear_power_interpolator : Callable
            Function of k and z returning the non-linear power spectrum.
        chi_min : float
            Minimum comoving distance for log-spaced array.
        chi_max : float
            Maximum comoving distance for log-spaced array.
        num : int, optional
            Number of points in log-spaced array. The default is 1000.
        ls_limber : int, optional
            Minimum multipole value for Limber approximation. The default is 200.
        miniter : int, optional
            Minimum number of iterations for the quadratue integration. The default is 1500.

        Returns
        -------
        result : array_like
            Array of projected power spectra.
        """

        linear_power_interpolator_redshift_zero = lambda k: linear_power_interpolator(0, k)
        diff_power_interpolator = lambda z, k, grid: non_linear_power_interpolator(z, k, grid = grid) - linear_power_interpolator(z, k, grid = grid)

        ells_non_limber = ls[ls < ls_limber]
        ells_limber = ls[ls >= ls_limber]

        chilog = np.logspace(np.log10(chi_min),np.log10(chi_max), num = num, endpoint = True)

        fftA = self.get_log_object_from_inputs_from_z_functions(chilog, zs, chis, hubble, window_A, growth, bias_A)
        fftB = self.get_log_object_from_inputs_from_z_functions(chilog, zs, chis, hubble, window_B, growth, bias_B)

        def get_element(ell):
            kA, FAk = fftA.fftlog(ell)
            kB, FBk = fftB.fftlog(ell)
            fA, fB = sinterp.interp1d(kA, FAk, fill_value = 0., bounds_error = False), sinterp.interp1d(kB, FBk, fill_value = 0., bounds_error = False)
            return self._integrate_fftlog(kA, fA, fB, linear_power_interpolator_redshift_zero, miniter)

        #for the first part use fftlog + limber
        result = np.array(list(map(get_element, ells_non_limber)))+super().integrate(ells_non_limber, hubble(zs), chis, window_A(zs), window_B(zs), diff_power_interpolator)

        #now for the rest of the modes, just use the Limber approximation
        result = np.append(result, super().integrate(ells_limber, hubble(zs), chis, window_A(zs), window_B(zs), non_linear_power_interpolator))



        return result

    
    def _integrate_fftlog(self, k, fftlog1, fftlog2, Plin, miniter = 1500):
        function = lambda k: fftlog1(k)*fftlog2(k)*Plin(k)*k**2
        a, b = k.min(), k.max()
        return sintegrate.quadrature(function, a, b, rtol = 1e-25, miniter = miniter)[0]*2/np.pi