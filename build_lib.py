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
import ubc_tbarpes.slab_lib as slib
import ubc_tbarpes.SK as SKlib
import ubc_tbarpes.klib as klib



###Build Basis
def gen_basis(basis,soc):
	
    bulk_basis = []
    for a in list(enumerate(basis['atoms'])):
        for o in list(enumerate(basis['orbs'][a[0]])):
            try:
                
                bulk_basis.append(olib.orbital(a[1],len(bulk_basis),o[1],basis['pos'][a[0]],basis['Z'][a[1]],orient=basis['orient'][a[0]][o[0]]))
            except KeyError:
                bulk_basis.append(olib.orbital(a[1],len(bulk_basis),o[1],basis['pos'][a[0]],basis['Z'][a[1]]))
    if soc['soc'] and not basis['slab']['bool']:
        bulk_basis = olib.spin_double(bulk_basis,soc['lam'])  
    basis['bulk'] = bulk_basis

    if basis['slab']['bool']:
        slab_obj = slib.slab(basis['slab']['hkl'],basis['slab']['cells'],basis['slab']['buff'],basis['slab']['avec'],bulk_basis,basis['slab']['term'])
        if soc['soc']:        
            slab_obj.slab_base = olib.spin_double(slab_obj.slab_base,soc['lam'])
        basis['slab']['obj'] = slab_obj
    
    return basis

###Generate a Kpath
def gen_K(Kdic,avec=None):
	#
    if Kdic['type']=='F':
        B = klib.bvectors(avec)
        klist = [np.dot(k,B) for k in Kdic['pts']]
    elif Kdic['type']=='A':
        klist = [k for k in Kdic['pts']]
    else:
        klist = []
        print('You have not entered a valid K path. Proceed with caution.')
    Kobj = klib.kpath(klist,Kdic['grain'],Kdic['labels'])

    return Kobj


###Built Tight Binding Model
def gen_TB(Bdict,H_args,Kobj):
    
    if not Bdict['slab']['bool']:
        TB = TBlib.TB_model(Bdict['bulk'],H_args,Kobj)
    else:
        if H_args['so']:
            H_args['so'] = False
            TB_o = TBlib.TB_model(Bdict['bulk'][:int(len(Bdict['bulk'])/2)],H_args,Kobj)
            H_args['so'] = True
        else:
            TB_o = TBlib.TB_model(Bdict['bulk'],H_args,Kobj)
        
        H_slab = Bdict['slab']['obj'].slab_TB(TB_o.mat_els,H_args,Kobj)
        Hnew_args = {'type':'list','list':H_slab,'cutoff':H_args['cutoff'],'renorm':H_args['renorm'],'offset':H_args['offset'],'tol':H_args['tol'],'so':H_args['so']}
        TB = TBlib.TB_model(Bdict['slab']['obj'].slab_base,Hnew_args,Kobj) 

    return TB


###Build ARPES Experiment



####Do experiments



