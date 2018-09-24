#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 11:15:35 2017

@author: ryanday

MIT License

Copyright (c) 2018 Ryan Patrick Day

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import ubc_tbarpes.orbital as olib
import ubc_tbarpes.TB_lib as TBlib
import ubc_tbarpes.H_library as Hlib
import ubc_tbarpes.slab as slib
import ubc_tbarpes.SK as SKlib
import ubc_tbarpes.klib as klib



###Build Basis
def gen_basis(basis):
	
    bulk_basis = []
    for a in list(enumerate(basis['atoms'])):
        for o in list(enumerate(basis['orbs'][a[0]])):
            try:
                
                bulk_basis.append(olib.orbital(a[1],len(bulk_basis),o[1],basis['pos'][a[0]],basis['Z'][a[1]],orient=basis['orient'][a[0]][o[0]]))
            except KeyError:
                bulk_basis.append(olib.orbital(a[1],len(bulk_basis),o[1],basis['pos'][a[0]],basis['Z'][a[1]]))
    if basis['spin']['bool']:
        bulk_basis = olib.spin_double(bulk_basis,basis['spin']['lam'])  
    basis['bulk'] = bulk_basis
    
    return basis

###Generate a Kpath
def gen_K(Kdic):
	#
    if Kdic['type']=='F':
        B = klib.bvectors(Kdic['avec'])
        klist = [np.dot(k,B) for k in Kdic['pts']]
    elif Kdic['type']=='A':
        klist = [k for k in Kdic['pts']]
    else:
        klist = []
        print('You have not entered a valid K path. Proceed with caution.')
    Kobj = klib.kpath(klist,Kdic['grain'],Kdic['labels'])

    return Kobj


###Built Tight Binding Model
def gen_TB(Bdict,H_args,Kobj,slab_dict=None):
    if type(slab_dict)==dict:
        if H_args['spin']['bool']:
            Bdict['bulk'] = Bdict['bulk'][:int(len(Bdict['bulk'])/2)]
            Hspin = True
            H_args['spin']['bool'] = False #temporarily forestall incorporation of spin
        else:
            Hspin=False
    TB = TBlib.TB_model(Bdict['bulk'],H_args,Kobj)
    if type(slab_dict)==dict:
        slab_dict['TB'] = TB
        TB,slab_H = slib.bulk_to_slab(slab_dict) 
        if Hspin:
            TB.basis = olib.spin_double(list(TB.basis),Bdict['spin']['lam']) 

        H_args['type']='list'
        H_args['list'] = slab_H
        
        H_args['avec'] = TB.avec
        H_args['spin']['bool']=Hspin

        TB.mat_els = TB.build_ham(H_args)
    return TB


###Build ARPES Experiment



####Do experiments



