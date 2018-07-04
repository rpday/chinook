# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 11:18:28 2018

@author: rday
"""

from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()
    f.close()
    
setup(name= 'ubc_tbarpes',
      version = '2.0',
      long_description=readme(),
      classifiers =[
              'Development Status :: Beta ',
              'License :: MIT License',
              'Programming Language :: Python :: 3.6', 
              'Topic :: Scientific :: Simulation ',
              ],
      author = 'Ryan P. Day', 
      author_email ='rday@phas.ubc.ca',
      license= 'MIT',
      packages= ['ubc_tbarpes'],
      install_requires=['numpy','matplotlib','scipy','sympy','operator'],
      include_package_data=True,
      package_data = {'ubc_tbarpes':['electron_configs.txt']}
      )
