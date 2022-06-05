# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 10:51:26 2019

@author: rday
"""

import numpy as np

import chinook.build_lib as build_lib
import chinook.ARPES_lib as arpes_lib


avec = np.array([[1.0,0.0,0.0],
                [0.0,1.0,0.0],
                [0.0,0.0,1.0]])

spin_args = {'bool':False}

basis_args = {'atoms':[],
         'Z':{},
         'pos':[],
         'orbs':[[]],
         'spin':spin_args}

hamiltonian_args = {'type':'SK',
         'V':{},
         'cutoff':1.0,
         'renorm':1.0,
         'offset':0.0,
         'tol':1e-4,
         'avec':avec,
         'spin':spin_args}

momentum_args= {'type':'F',
                'avec':avec,
                'grain':100,
                'pts':[]}

basis = build_lib.gen_basis(basis_args)
kpath = build_lib.gen_K(momentum_args)
TB = build_lib.gen_TB(basis,hamiltonian_args,kpath)

TB.solve_H()
TB.plotting()


####CALCULATION OF ARPES INTENSITY ######

#arpes_args={'cube':{'X':[],'Y':[],'kz':0.0,'E':[]},
#            'SE':['constant',0.001],
#            'hv': 21.2,
#            'pol':np.array([1,0,0]),
#            'resolution':{'E':0.01,'k':0.01},
#            'T':4.2}
#
#experiment = arpes_lib.experiment(TB,arpes_args)
#experiment.datacube()
#experiment.spectral()

