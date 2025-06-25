# setup.py
from setuptools import setup, find_packages

setup(
    name='simple_hil_gym',
    version='0.0.1',
    packages=find_packages(),  # This finds all packages (folders with __init__.py)
    install_requires=['gymnasium', 'numpy', 'pygame'],
)