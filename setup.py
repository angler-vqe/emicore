#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
    name="src",
    use_scm_version=True,
    packages=find_packages(include=['src*']),
    install_requires=[
        'torch>=1.5.0',
        'numpy',
        'tqdm',
        'matplotlib',
        'click',
        'tensorboard',
        'joblib',
        'scipy',
        'noisyopt',
        'scikit-learn',
        'qiskit',
        'h5py',
        'aim'
    ],
    setup_requires=[
        'setuptools_scm',
    ],
)
