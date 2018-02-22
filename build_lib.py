#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 11:15:35 2017

@author: ryanday
"""
import numpy as np
import orbital as olib
import TB_lib as TBlib
import H_library as Hlib
import slab_lib as slib
import SK as SKlib
import klib as klib



###Build Basis
def gen_basis(basis,soc):
	
    bulk_basis = []
    for a in list(enumerate(basis['atoms'])):
        for o in basis['orbs'][a[0]]:
            bulk_basis.append(olib.orbital(a[1],len(bulk_basis),o,basis['pos'][a[0]],basis['Z'][a[1]]))
    if soc['soc'] and not basis['slab']['bool']:
        bulk_basis = olib.spin_double(bulk_basis,soc['lam'])  
    basis['bulk'] = bulk_basis

    if basis['slab']['bool']:
        print('yes')
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



