#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 20:23:02 2019

@author: prmiles
"""


def gsas_install_display(module):
    print('When importing {} a "ModuleNotFoundError" occured. \n'.format(
            module)
          + 'This usually occurs when GSAS-II has not been installed\n'
          + 'or it has not been correctly added to the Python search '
          + 'path.')
    print('For details on how to install GSAS-II, please refer to the '
          + 'software homepage:\n\t'
          + 'https://subversion.xray.aps.anl.gov/trac/pyGSAS')
    print('To add GSAS-II to the Python search path, appending the following'
          + 'to your script:'
          + '\n\timport sys'
          + '\n\tsys.path.append("<path>/GSASII")'
          + '\n\t# May need to append one or both of these paths as well'
          + '\n\tsys.path.append("<path>/GSASII/fsource")'
          + '\n\tsys.path.append("<path>/GSASII/bindist")')
