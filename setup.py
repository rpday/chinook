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
    
setup(name= 'chinook',
      version = '1.0.0',
      author = 'Ryan P. Day', 
      author_email = 'ryanday7@gmail.com',
      description = 'Tools for tight-binding and simulation of ARPES',
      long_description=readme(),
      url = 'https://github.com/rpday/chinook',
      classifiers =[
		'Development Status :: 4 - Beta',              
		'License :: OSI Approved :: MIT License',              	
		'Programming Language :: Python :: 3 :: Only',             
        	'Topic :: Scientific/Engineering :: Physics',
          'Operating System :: OS Independent'
	      ],

      license= 'MIT',
      packages= ['chinook'],
      install_requires=['numpy','matplotlib','scipy'],
      include_package_data=True,
      package_data = {'chinook':['atomic_mass.txt','electron_configs.txt']}
      )
