import jax.numpy as np

from typing import Callable, List, Tuple, Set, Union

import itertools

from . import cosmoconstants, projector


class LimberProjector(projector.Projector):


    def __init__(self, zs: np.ndarray, spectra: List[Tuple[str, Callable]], ls: np.ndarray, kmax: float, cSpeedKmPerSec: float = cosmoconstants.CSPEEDKMPERSEC, **kwargs):
        super().__init__(zs = zs, spectra = spectra, **kwargs)
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
        #actually it should check both excluded windows and excluded windows combinations at the same time for compatibility
        selected_windows = [window for window in self.windows if window not in excluded_windows]
        Nw = len(selected_windows)
        allcombs = list(itertools.combinations_with_replacement(selected_windows, 2))
        allcombs = [combination for combination in allcombs if set(combination) not in excluded_windows_combinations]

        #Set up the result
        result = {}
        window_products = np.zeros((len(self.zs), Nw, Nw))
        spectra = []
        Nmaxspectra = 0
        #Make calculation
        for couple in allcombs:
            A, B = couple
            indA, indB = selected_windows.index(A), selected_windows.index(B)
            type_A, window_A = getattr(self, A)

            if B != A:
                type_B, window_B = getattr(self, B)
                #type_AB = type_A + type_B #if type_A != type_B else type_A
            else:
                type_B = type_A
                window_B = window_A

            type_AB = type_A + type_B
            spectrum = self.spectrum(type_AB) #spectrum here can be a callable or a list of callables

            Nspectra = len(spectrum)
            Nmaxspectra = Nspectra if Nspectra > Nmaxspectra else Nmaxspectra

            spectra += [(indA, indB, type_AB, spectrum)]

            window_product = window_A*window_B
            window_products[:, indA, indB] = window_product

        ls = self.ls        
        resultmatrix = self.integrate(ls, self.zs, self.ws, Hzs, chis, window_products, spectra, Nmaxspectra)

        for couple in allcombs:
            A, B = couple
            indA, indB = selected_windows.index(A), selected_windows.index(B)
            result[couple] = resultmatrix[:, indA, indB]

        self.resultmatrix = resultmatrix
        self.currentcombs = allcombs
        self.currentwindows = selected_windows
        self.Nmaxspectra = Nmaxspectra

        results = self.get_results_object(result)
        return results

    def get_results_object(self, resultdictionary):
        return projector.Results(self.ls, resultdictionary)
    
    def update_results_with_biases(self, newbiases: np.ndarray):
        '''
        Update the results with new biases
        '''
        #resultsmatrix has self.ls elements, Nw windows, Nw windows, Nspectra
        assert self.resultmatrix.shape[1:] == newbiases.shape, "The number of biases should be compatible with the existing results"
        self.resultmatrix = np.einsum("abcd, bcd -> abc", self.resultmatrix, newbiases)

        result = {}
        for couple in self.currentcombs:
            A, B = couple
            indA, indB = self.currentwindows.index(A), self.currentwindows.index(B)
            result[couple] = self.resultmatrix[:, indA, indB]

        results = self.get_results_object(result)

        return results

    def integrate(self, ls, zs, ws, Hzs, chis, window_product, power_interpolator, Nmaxspectra:int, chiint: bool = False):
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
        
        if chiint:
            common_prefactor = Hzs**2./chis/chis/self.cSpeedKmPerSec**2.
        else:
            #zint
            common_prefactor = Hzs*1/self.cSpeedKmPerSec*1./chis/chis

        cl = np.array([self._integrate(l = l, interpolator = power_interpolator, zs = zs, ws = ws, chis = chis, window_product = window_product, common_prefactor = common_prefactor, kmax = self.kmax, Nmaxspectra = Nmaxspectra) for l in ls])
        return cl

    #@staticmethod
    def _integrate(self, l: np.ndarray, interpolator: Callable, zs: np.ndarray, ws: np.ndarray, chis: np.ndarray, window_product: np.ndarray, common_prefactor: np.ndarray, kmax: float, Nmaxspectra:int, kmin: float = 1e-4):
        '''
        For now assumes scipy interpolator, might change in the future

        Based on CAMB demo code and orphics code.

        TODO: Improve sampling, based on redshift distribution.
        '''

        zmin = 0.

        #_window_for_calculations = np.ones(chis.shape) #this is just used to set to zero k values out of range of interpolation
        k = (l+0.5)/chis
        selection = (k >= kmin) & (k <= kmax)
        _window_for_calculations = selection*1

        zsel, ksel = zs[selection], k[selection]

        #calculate power only for the modes that respect kmin, kmax constraints
        power = np.zeros((*window_product.shape, Nmaxspectra)) #np.zeros_like(chis)
        spectra_results = {}
        #this thing better to make it an object that manages the spectra
        #something useful also is to make it always remember the spectra, and common quantities
        #e.g. g1g1, g2g2, and linear bias: so b1^2*Pmm, b2^2*Pmm 
        for (indA, indB, type_, spectrum) in interpolator:
            if type_ not in spectra_results.keys():
                if type(spectrum) is list:
                    #take cosmo interpolator calculate at selected zsel
                    #now you have to select for all the ksel
                    #for each zsel you want to interpolate at ksel
                    result = np.array([spectrum_i(zsel, ksel, grid = False) for spectrum_i in spectrum])
                else:
                    result = spectrum(zsel, ksel, grid = False)
                spectra_results[type_] = result
            else:
                result = spectra_results[type_]
            power[selection, indA, indB, :] = result.T
            
        #power[selection] = interpolator(zs[selection], k[selection], grid = False) 
    
        common = np.einsum("abcd, a -> abcd", power, ((_window_for_calculations)*common_prefactor))

        #integration routine here     
        estCl = np.einsum("a, abcd, abc -> bcd", ws[zs >= zmin], common[zs >= zmin], (window_product)[zs >= zmin])
        
        return estCl



