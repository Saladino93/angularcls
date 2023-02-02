import numpy as np

from typing import Callable, List, Tuple, Set, Union

import itertools

import cosmoconstants

import projector



class LimberProjector(projector.Projector):


    def __init__(self, zs: np.ndarray, spectra: List[Tuple[str, Callable]], ls: np.ndarray, kmax: float, cSpeedKmPerSec: float = cosmoconstants.CSPEEDKMPERSEC):
        super().__init__(zs = zs, spectra = spectra)
        self.ls = ls
        self.kmax = kmax
        self.cSpeedKmPerSec = cSpeedKmPerSec


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

    def integrate(self, ls, Hzs, chis, window_A, window_B, power_interpolator):
        '''
        Parameters
        ----------
        ls: np.ndarray
            The multipoles at which to evaluate the projected power
        power_interpolator: Callable
            The power interpolator for the fields, P_{XY}

        Returns
        -------

        '''
        cl = []
        
        #Common factor to the windows in the Limber integrand
        
        common_prefactor = Hzs**2./chis/chis/self.cSpeedKmPerSec**2.
        window_product = window_A*window_B

        #CHECK THIS STEP, ESPECIALLY FOR LENSING
        dchis = (chis[2:]-chis[:-2])/2
        chis = chis[1:-1]
        window_product = window_product[1:-1]
        Hzs = Hzs[1:-1]##
        zs = self.zs[1:-1]
        common_prefactor = common_prefactor[1:-1]

        cl = np.array([self._integrate(l = l, interpolator = power_interpolator, zs = zs, chis = chis, dchis = dchis, window_product = window_product, common_prefactor = common_prefactor, kmax = self.kmax) for l in ls])
        return cl

    @staticmethod
    def _integrate(l: np.ndarray, interpolator: Callable, zs: np.ndarray, chis: np.ndarray, dchis: np.ndarray, window_product: np.ndarray, common_prefactor: np.ndarray, kmax: float):
        '''
        For now assumes scipy interpolator, might change in the future
        '''

        zmin = 0.

        _window_for_calculations = np.ones(chis.shape) #this is just used to set to zero k values out of range of interpolation
        k = (l+0.5)/chis
        _window_for_calculations[k < 1e-4]=0
        _window_for_calculations[k >= kmax]=0

        power = interpolator(zs, k, grid = False)
        
        common = ((_window_for_calculations*power)*common_prefactor)[zs >= zmin]   

        #integration routine here     
        estCl = np.dot(dchis[zs >= zmin], common*(window_product)[zs >= zmin])
        
        return estCl


