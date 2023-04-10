"""
Define fields to be cross-correlated.
"""

import numpy as np

from typing import Callable


class Field(object):

    """
    Class to define a field to be cross-correlated.
    """

    def __init__(self, name: str, window_function: Callable, window_type: str = 'm'):
        """
        Parameters
        ----------
        name : str
            Name of the field
        window_function : Callable
            Function that returns the window function for a given redshift
        window_type : str, optional
            Type of window function. 'm' for multiplicative, 'a' for additive, by default 'm'
        """
        self.name = name
        self.window_function = window_function
        self.window_type = window_type

    def __call__(self, zs: np.ndarray) -> np.ndarray:
        return self.window_function(zs)