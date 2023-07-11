#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
    name='emicore',
    use_scm_version=True,
    packages=find_packages(where='src', include=['emicore*']),
    package_dir={'': 'src'},
    install_requires=[
        'torch>=1.5.0',
        'numpy',
        'scipy',
        'qiskit==0.42.1',
        'click',
        'h5py',
        'tqdm',
    ],
    setup_requires=[
        'setuptools_scm',
    ],
)
