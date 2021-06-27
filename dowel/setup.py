import os

from setuptools import find_packages
from setuptools import setup

required = [
    # Please keep alphabetized
    'matplotlib',
    'numpy',
    'python-dateutil',
    'scipy',
    'tabulate',
    'tensorboardX',
]

setup(
    name='dowel',
    packages=find_packages(where='.'),
    package_dir={'': '.'},
    python_requires='>=3.5',
    install_requires=required,
)
