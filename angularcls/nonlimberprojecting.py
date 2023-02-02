import limberprojecting

import numpy as np

import scipy.integrate as sintegrate

import scipy.interpolate as sinterp

import fftlog

import cosmoconstants

from typing import Callable, Union


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

    def integrate(self, ls: np.ndarray, zs: np.ndarray, chis: np.ndarray, hubble: Callable, growth: Callable, bias_A: Callable, bias_B: Callable, window_A: Callable, window_B: Callable, linear_power_interpolator: Callable, diff_power_interpolator: Callable, chi_min: float, chi_max: float, num: int = 1000, equal: bool = False):
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
            Function of k and z returning the linear power spectrum at z = 0.
        diff_power_interpolator : Callable
            Function of k and z returning the difference between linear and non-linear power spectrum.
        chi_min : float
            Minimum comoving distance for log-spaced array.
        chi_max : float
            Maximum comoving distance for log-spaced array.
        num : int, optional
            Number of points in log-spaced array. The default is 1000.
        equal : bool, optional
            If True A = B and we do fftlog only once. The default is False.
        
        Returns
        -------
        result : array_like
            Array of projected power spectra.
        """

        chilog = np.logspace(np.log10(chi_min),np.log10(chi_max), num = num, endpoint = True)

        fftA = self.get_log_object_from_inputs_from_z_functions(chilog, zs, chis, hubble, window_A, growth, bias_A)
        fftB = self.get_log_object_from_inputs_from_z_functions(chilog, zs, chis, hubble, window_B, growth, bias_B) if not equal else fftA

        def get_element(ell):
            kA, FAk = fftA.fftlog(ell)
            kB, FBk = fftB.fftlog(ell) if not equal else (kA, FAk)
            fA, fB = sinterp.interp1d(kA, FAk, fill_value = 0., bounds_error = False), sinterp.interp1d(kB, FBk, fill_value = 0., bounds_error = False)
            return self._integrate_fftlog(kA, fA, fB, linear_power_interpolator)

        result = np.array(list(map(get_element, ls)))

        return result



    def obtain_spectra(self, Hubble: Union[Callable, np.ndarray], chi: Union[Callable, np.ndarray],
                       excluded_windows: List[str] = [], excluded_windows_combinations: List[Set[str]] = []) -> projector.Results:
        '''
        Note the 
        '''

        Hzs = Hubble(self.zs)
        chis = chi(self.zs)

        #Select windows needed for the calculation
        selected_windows = [window for window in self.windows if window not in excluded_windows]
        allcombs = list(itertools.combinations_with_replacement(selected_windows, 2))
        allcombs = [combination for combination in allcombs if set(combination) not in excluded_windows_combinations]

        #Set up the result
        result = {}
        #Make calculation
        for couple in allcombs:
            A, B = couple
            type_A, window_A = getattr(self, A)
            if B != A:
                type_B, window_B = getattr(self, B)
                type_AB = type_A + type_B if type_A != type_B else type_A
                spectrum = self.spectrum(type_AB)
            else:
                spectrum = self.spectrum(type_A)
                window_B = window_A

            ls = self.ls
            result[couple] = self.integrate(ls, Hzs, chis, window_A, window_B, spectrum)

        results = projector.Results(ls, result)
        return results
    
    def _integrate_fftlog(self, k, fftlog1, fftlog2, Plin):
        function = lambda k: fftlog1(k)*fftlog2(k)*Plin(k)*k**2
        a, b = k.min(), k.max()
        return sintegrate.quadrature(function, a, b, rtol = 1e-25, miniter = 200)[0]*2/np.pi