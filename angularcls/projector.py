import numpy as np

from typing import Callable, List, Tuple, Union, Dict

import abc

class Projector(object):

    def __init__(self, zs: np.ndarray, spectra: List[Tuple[str, Callable]]):
        '''
        Parameters
        ----------
        spectra: List[Tuple[str, Callable]]
            A list of tuples, where the first key indicates the type of the spectrum, while the second a Callable. 
            Once you have more than one key, it is imperative that you indicate also their combinations, with None a valid spectrum 'interpolator'.
        
        Examples
        --------
        >> spectra = [('m', P_mm_interpolator), ('g', P_gg_interpolator), ('mg', P_mg_interpolator)]
        '''

        self.zs = zs

        self._window_list = []
        #TO DO, CHECK COMBINATIONS OF SPECTRA ARE PRESENT TOO
        self._spectra = {element[0]: element[1] for element in spectra}
        self.spectra_keys = [element[0] for element in spectra]

    def spectrum(self, key: str) -> Callable:
        try:
            result = self._spectra[key]
        except:
            result = self._spectra[key[::-1]]
        return result

    @property
    def windows(self):
        return self._window_list

    def _update_windows_list(self, new_name: str):
        self._window_list += [new_name]

    def update_window(self, window_name: str, window_properties: Tuple[str, Union[Callable, np.ndarray]]):
        '''
        Examples
        --------
        >> window_name = 'k'
        >> window_properties = ('m', kappa_window_function)
        '''
        _window_prop = window_properties[1]
        window_values = _window_prop(self.zs) if type(_window_prop) is Callable else _window_prop
        window_properties = (window_properties[0], window_values)
        setattr(self, window_name, window_properties)
        if window_name not in self.windows:
            self._update_windows_list(window_name)


    @abc.abstractmethod
    def obtain_spectra(self):
        return

    @abc.abstractmethod
    def integrate(self):
        return 


class Results(object):
    def __init__(self, ls: np.ndarray, result_dict: Dict):
        self.results = result_dict
        self.ls = ls
        #fig, ax = plt.subplots(nrows = len(result_dict.keys()))
        #self.fig = fig
        #self.ax = ax

    def get(self, keyA: str, keyB: str = '') -> np.ndarray:
        if keyB == '':
            keyB = keyA
        try:
            result = self.results[(keyA, keyB)]
        except:
            result = self.results[(keyB, keyA)]
        return result

    def get_big_data_vector(self) -> np.ndarray:
        return list(self.results.keys()), np.hstack(list(self.results.values()))