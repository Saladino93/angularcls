"""
Define fields to be cross-correlated.
"""

import jax.numpy as jnp

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
        self.window_function = jnp.jit(window_function)
        self.window_type = window_type

    def __call__(self, zs: jnp.ndarray) -> jnp.ndarray:
        return self.window_function(zs)
    

class Galaxy(Field):
    """
    Galaxy field with some properties to be defined.
    """
    def __init__(self, b1: jnp.float, b2: jnp.float = 0., bs2: jnp.float = 0., b3nl: jnp.float = 0., bk2: jnp.float = 0., **kwargs):
        self().__init__(**kwargs)

        self.b1 = b1
        self.b2 = b2
        self.b3nl = b3nl
        self.bs2 = bs2
        self.bk2 = bk2

    