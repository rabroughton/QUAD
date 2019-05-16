Quantitative Uncertainty Analysis for Diffraction (QUAD)
========================================================

This is a research tool that allows analysis of X-ray and neutron
diffraction data to infer the structure of materials with quantifiable 
uncertainty. QUAD uses Bayesian statistics and Markov chain sampling 
algorithms, together with components from the open source GSAS-II package, 
to create posterior probability distributions on all material structure 
parameters modeled by researchers.

While the tool has been used for several research projects, a formal Python package is still under development.  A useable version of the package is expected to be available by May 31, 2019.

Installation
============

The package can be installed by executing the following command in a terminal

::

    pip install git+https://github.com/rabroughton/QUAD.git
   
The package requires the GSAS-II software. Installation instructions can be found at the `GSAS-II home page <https://subversion.xray.aps.anl.gov/trac/pyGSAS>`_.

For use in a python script, the GSAS-II path must be appended. 

::

    import sys
    sys.path.append('<path>/GSASII')
    # May need to append one or both of these paths
    sys.path.append('<path>/GSASII/fsource')
    sys.path.append('<path>/GSASII/bindist')

**Authors:** Susheela Singh, Christopher M. Fancher, Alyson Wilson, Brian Reich, 
Zhen Han, Ralph C. Smith, and Jacob L. Jones

**Maintainers:** Rachel A. Broughton and Paul R. Miles

**Funding:**
  * NSF: DMR-1445926 (RADAR Project ID 2014-2831)
  * Consortium for Nonproliferation Enabling Capabilities [Department of Energy, National Nuclear Security Administration]: DE-NA0002576 (RADAR Project ID 2014-0501)

**Acknowledgement:** This product includes software produced by UChicago Argonne, LLC 
under Contract No. DE-AC02-06CH11357 with the Department of Energy.

**License:**

  * `NCSU`_

.. _NCSU: license.txt
