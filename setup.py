#setup file

from setuptools import setup, find_packages

setup(
    name = 'angularcls',
    version = '0.0.1',
    description = 'Angular power spectra for cosmology.',
    url = 'https://github.com/Saladino93/angularcls'
    packages = ['angularcls'],
    package_dir = {'angularcls': 'angularcls'}
    )


