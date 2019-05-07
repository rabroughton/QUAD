from setuptools import setup
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
      author_email = 'rabrough@ncsu.edu'
      )
