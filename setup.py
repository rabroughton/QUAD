from setuptools import setup, find_packages
import os
import re

def get_version():
    VERSIONFILE = os.path.join('QUAD', '__init__.py')
    with open(VERSIONFILE, 'rt') as f:
        lines = f.readlines()
    vgx = '^__version__ = \"\d+\.\d+\.\d.*\"'
    for line in lines:
        mo = re.search(vgx, line, re.M)
        if mo:
            return mo.group().split('"')[1]
    raise RuntimeError('Unable to find version in %s.' % (VERSIONFILE,))

setup(
      name='QUAD',
      version=get_version(),
      description=('Bayesian computing of material parameters from diffraction patterns'),
      url='https://github.com/rabroughton/QUAD',
      download_url='https://github.com/rabroughton/QUAD',
      author='Rachel Broughton',
      author_email = 'rabrough@ncsu.edu',
      package_dir={'QUAD': 'QUAD'},
      packages=find_packages(),
      zip_safe=False,
      install_requires=['scipy>=1.0', 'statsmodels>=0.9.0', 'matplotlib>=2.2.0', 'bspline>=0.1.1', 'seaborn>=0.9.0],
      )
