#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 12:17:01 2017

@author: ryanday
"""
import sys
sys.path.append('C:/Users/rday/Documents/TB_ARPES/2018/TB_ARPES_2018/TB_ARPES-master/')

import ubc_tbarpes.build_lib as build_lib
import numpy as np

import ubc_tbarpes.ARPES_lib as ARPES
import scipy.ndimage as nd
import ubc_tbarpes.operator_library as ops
import ubc_tbarpes.plt_sph_harm as sph
import ubc_tbarpes.direct as direct
import ubc_tbarpes.direct as integral
import ubc_tbarpes.slab as slib
import ubc_tbarpes.H_library as hlib
import ubc_tbarpes.TB_lib as TB_lib

####THis Hamiltonian model has elements where j<i, this is proving to be a funny problem when generating the slab Hamiltonian
## I should check this Hamiltonian then...I'm probably ignoring these matrix elements, which may have caused some issues.

if __name__=="__main__":
    a,c =  3.7734,5.5258 
   # a,c = 5*np.sqrt(2),10
    avec = np.array([[a/np.sqrt(2),a/np.sqrt(2),0.0],[-a/np.sqrt(2),a/np.sqrt(2),0.0],[0.0,0.0,c]])
    
    filenm ='FeSe_o.txt'#'garbage.txt'# 
    CUT,REN,OFF,TOL=a*3,1/1.4,0.12,0.001
#    CUT,REN,OFF,TOL=a*10,1,0.015,0.001
#    CUT,REN,OFF,TOL=a*10,1,0.005,0.001
    G,X,M,Z,R,A = np.array([0,0,0]),np.array([0,-0.5,0]),np.array([0.5,-0.5,0]),np.array([0,0,0.5]),np.array([0.5,0,0.5]),np.array([0.5,-0.5,0.5])

#    pt1,pt2,pt3 = np.zeros(3),np.array([0.2119,0.2119,0]),np.array([0.2119,-0.2119,0])

    spin = {'soc':False,'lam':{0:0.03}}
    
    slab_dict = {'bool':False,
                'hkl':np.array([0,0,1]),
                'cells':20,
                'buff':8,
                'term':0,
                'avec':avec}

    Bd = {'atoms':[0,0],
			'Z':{0:26},
			'orbs':[["32xy","32XY","32xz","32yz","32ZR"],["32xy","32XY","32xz","32yz","32ZR"]],
			'pos':[np.array([-a/np.sqrt(8),0,0]),np.array([a/np.sqrt(8),0,0])],
            'slab':slab_dict}

    Kd = {'type':'F',
			'pts':[M,G,Z],
			'grain':200,
			'labels':['M','$\Gamma$','M','X','$\Gamma$']}


    Hd = {'type':'txt',
			'filename':filenm,
			'cutoff':CUT,
			'renorm':REN,
			'offset':OFF,
			'tol':TOL,
			'so':spin['soc']}
 
    	#####
    Bd = build_lib.gen_basis(Bd,spin)
    Kobj = build_lib.gen_K(Kd,avec)
    TB = build_lib.gen_TB(Bd,Hd,Kobj)
    
#    dB = -0.0089
    dS = -0.015#0.0096
#    for ii in range(len(TB.mat_els)):
#        if TB.mat_els[ii].i==TB.mat_els[ii].j:
#            if TB.mat_els[ii].i%5==2:
#                TB.mat_els[ii].H.append([0,0,0,dS])
#            elif TB.mat_els[ii].i%5==3:
#                TB.mat_els[ii].H.append([0,0,0,-dS])
##                
#        if (TB.mat_els[ii].i%10==2 and TB.mat_els[ii].j%10==7) or (TB.mat_els[ii].i%10==3 and TB.mat_els[ii].j%10==8):
#            TB.mat_els[ii].H.append([a/np.sqrt(2),0,0,dB])
#            TB.mat_els[ii].H.append([-a/np.sqrt(2),0,0,dB])
#            TB.mat_els[ii].H.append([0,a/np.sqrt(2),0,-dB])
#            TB.mat_els[ii].H.append([0,-a/np.sqrt(2),0,-dB])
#            if ii<len(TB.basis)/2:
#                TB.mat_els[ii].H.append([0,0,0,1.0e-9])
#            else:
#                TB.mat_els[ii].H.append([0,0,0,-1.0e-9])
#                
    TB.solve_H()
    TB.plotting()

#    
    miller = np.array([0,0,1])
    EV = TB.Evec
    EB = TB.Eband
    tmp_H = TB.mat_els
    new_basis,vn_b,R = slib.gen_surface(avec,miller,TB.basis)
    H_surf = slib.H_surf(new_basis,vn_b,TB.mat_els,R)
    nvec,cull=slib.gen_slab(new_basis,vn_b,50,30,[2,3])
#
#
    basis = TB.basis
    TB.basis = new_basis
    TB.mat_els = H_surf
    Kr = np.dot(Kobj.pts,R)
    Kd = {'type':'A',
			'pts':[Kr[0],Kr[1],Kr[2]],
			'grain':200,
			'labels':['M','$\Gamma$','M','X','$\Gamma$']}
    
    TB.Kobj= build_lib.gen_K(Kd,vn_b)
    
    TB.solve_H()
    TB.plotting()
###    
    mint,minb = 40,15
    term = (0,0)
    av_slab,slab_basis = slib.gen_slab(TB.basis,vn_b,mint,minb,term)
    Hslab = slib.build_slab_H(TB.mat_els,slab_basis,TB.basis,vn_b)
    
    TB.basis = slab_basis
    TB.mat_els = Hslab
    TB.solve_H()
    TB.plotting()
#    ktuple = ((1.0,1.5,300),(-.3,0.3,300),0.0)
#    Ef = 0.01
#    tol=0.0008
#    sig = integral.optical_conductivity(TB,avec,40,10)

#    FS = ops.FS(TB,ktuple,Ef,tol)
#    O = ops.LdotS(TB,axis='z',vlims=(-0.5,0.5),Elims=(-0.25,0.1))
#    jdos = integral.path_dir_op(TB,Kobj,hv=0.36,width=0.01,T=10)
####    
##    Ix = integral.plot_jdos(Kobj,TB,jdos,np.array([1,0,0]))
#    Iy = integral.plot_jdos(Kobj,TB,jdos,np.array([0,1,0]))
##    Iz = integral.plot_jdos(Kobj,TB,jdos,np.array([0,0,1]))
######    
#    Ixdos = direct.k_integrated(Kobj,TB,0.02,Iy)
#    Iydos = direct.k_integrated(Kobj,TB,0.02,Iy)
#    Izdos = direct.k_integrated(Kobj,TB,0.02,Iz)
#    
    
#    d1d,d1u = {o.index:o.proj for o in TB.basis[:5]},{(o.index-5):o.proj for o in TB.basis[10:15]}
#    d2d,d2u = {(o.index-5):o.proj for o in TB.basis[5:10]},{(o.index-10):o.proj for o in TB.basis[15:]}
#    d1 = {**d1d,**d1u}
#    d2 = {**d2d,**d2u}
#    ind =11
##    for oind in range(7,13):
##        ind = oind
#    psi_11 = np.array([list(TB.Evec[i,:5,ind])+list(TB.Evec[i,10:15,ind]) for i in range(len(Kobj.kpts))])
#    
#    psi_12 = np.array([list(TB.Evec[i,5:10,ind])+list(TB.Evec[i,15:,ind]) for i in range(len(Kobj.kpts))])
#       
#    #    psi_1 = np.array([list(TB.Evec[i,:,ind+2]) for i in range(len(Kobj.kpts))])
#    #    df = {o.index:o.proj for o in TB.basis}
#    #    psi_11 = np.array([list(TB.Evec[i,:5,ind:ind+2])+list(TB.Evec[i,10:15,ind:ind+2]) for i in range(len(Kobj.kpts))])
#        
#    for ki in range(len(Kobj.kpts)):
#        tmp =ki
##        strnm = 'C:\\Users\\rday\\Documents\\TB_ARPES\\2018\\TB_ARPES_2018\\plots\\FeSe_kz\\no_SO_{:d}_{:0.1f}Z.png'.format(ind,tmp)
#        sph.gen_psi(30,psi_11[ki],d1)#strnm)

#    O = ops.fatbs(proj,TB,vlims=(0,1),Elims=(-1,1),degen=True)
    
#    
#    
#    xmin,xmax,Nx=-0.25,0.25,60
#    dx = (xmax-xmin)/Nx
#    
#    ARPES_dict={'cube':{'X':[xmin,xmax,Nx],'Y':[-4*dx,4*dx,9],'kz':0.0,'E':[-0.1,0.05,100]},
#                'SE':[0.002],
#                'directory':'C:\\Users\\rday\\Documents\\TB_ARPES\\2018\\TB_ARPES_2018\\FeSe',
#                'hv': 37,
#                'pol':np.array([0,1,0]),
#                'mfp':7.0,
#                'resolution':{'E':0.005,'k':0.01},
#                'T':[True,10.0],
#                'W':4.0,
#                'angle':0,
#                'spin':None,
#                'slice':[False,-0.005]}
#                #'Brads':{'0-3-2-1':100.0,'0-3-2-3':0.0}}
#
##
##    
#
#    expmt = ARPES.experiment(TB,ARPES_dict)
#    expmt.datacube(ARPES_dict)
#    expmt.plot_slice(ARPES_dict)
#    Io_oo,Igo_oo = expmt.spectral(ARPES_dict,(1,4))
###    
##    
###    expmt.plot_gui(ARPES_dict)
##
##
##    
##    
##	#####