# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 10:14:10 2019

@author: rday
"""

import chinook.build_lib as build_lib
import chinook.ARPES_lib as arpes_lib
import numpy as np



avec = np.array([[1,0,0],[0,1,0],[0,0,1]])
CUT,REN,OFF,TOL = 10,1.0,0.0,0.001

basis = {'atoms':[],
         'Z':{},
         'pos':[],
         'orbs':[]}

momentum = {'type':'F',
            'avec':avec,
            'pts':[],
            'grain':100,
            'labels':[]}

hamiltonian = {'type':'',
               'avec':avec,
               'cutoff':CUT,
               'renorm':REN,
               'offset':OFF,
               'tol':TOL}


arpes = {'cube':{'X':[-1,1,50],'Y':[-1,1,50],'kz':0.0,'E':[-1,0.1]},
        'SE':['const',0.005],
        'hv':21.2,
        'pol':np.array([1,0,0]),
        'T':4.2,
        'resolution':{'E':0.01,'k':0.01}}


basis = build_lib.gen_basis(basis)
TB = build_lib.gen_TB(basis,hamiltonian,Kobj=build_lib.gen_K(momentum))

TB.solve_H()
TB.plotting()
#
#experiment = arpes_lib.experiment(TB,arpes)
#experiment.datacube()
#experiment.spectral(slice_select=('w',0))

