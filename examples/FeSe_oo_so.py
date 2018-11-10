#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 12:17:01 2017

@author: ryanday
FeSe: SOC and Orbital ORDER
"""
import sys
sys.path.append('C:/Users/rday/Documents/TB_ARPES/2018/TB_ARPES_2018/TB_ARPES-master/')

import ubc_tbarpes.build_lib as build_lib
import numpy as np
import matplotlib.pyplot as plt

import ubc_tbarpes.ARPES_lib as ARPES
import ubc_tbarpes.TB_lib as TB_lib
import ubc_tbarpes.H_library as Hlib
import ubc_tbarpes.operator_library as ops
import ubc_tbarpes.plt_sph_harm as sph
import ubc_tbarpes.direct as direct
import ubc_tbarpes.direct as integral


def redo_onsite(Hobj,off,ds):
    onsite = np.ones(20)*(-off)
    onsite[[2,7,12,17]]-=ds
    onsite[[3,8,13,18]]+=ds
    print(onsite)
    for hi in Hobj:
        if hi.i==hi.j:# find diagonals terms
            for hiH in hi.H:
                if np.linalg.norm(np.array(hiH[:3]))<1e-5:

                    hiH[-1]+=onsite[hi.i]
    return Hobj

def unpack(H):
    '''
    Reduce a Hamiltonian object down to a list of matrix elements. Include the Hermitian conjugate terms
    args:
        H -- Hamiltonian object
    return
        Hlist -- list of Hamiltonian matrix elements
    '''
    Hlist =[]
    for hij in H:
        for el in hij.H:
            Hlist.append([hij.i,hij.j,*el])
    return Hlist

def shift_H(Nb,off):
    tmp_H = [[i,i,0,0,0,off] for i in range(Nb)]

    return tmp_H

def redo_SO(basis,lamdict):
    for bi in basis:
        bi.lam = lamdict[bi.atom]
    
    return basis


def redo_OO(matels,dS):   
    for hi in matels:
        if hi.i==hi.j:
            if np.mod(hi.i,5)==2:
                hi.H.append([0,0,0,-dS])
            elif np.mod(hi.i,5)==3:
                hi.H.append([0,0,0,dS])
    
    return matels
#    
#def rebuild_H(Ho,basis,lamdict,dS,off):
#    Hnew = Ho + shift_H(len(basis),off) + redo_SO(basis,lamdict) + redo_OO(dS)
#    return TB_lib.gen_H_obj(Hnew)




if __name__=="__main__":
    a,c =  3.7734,5.5258 
    avec = np.array([[a/np.sqrt(2),a/np.sqrt(2),0.0],[-a/np.sqrt(2),a/np.sqrt(2),0.0],[0.0,0.0,c]])
    Fe1,Fe2 = np.array([-a/np.sqrt(8),0,0]),np.array([a/np.sqrt(8),0,0])


    filenm = 'FeSe_BMA_MOD.txt'
    CUT,REN,TOL=a*4,1.0,1e-7
    OFF = 0.015
#    CUT,REN,OFF,TOL=a*3,1/1.4,0.12,0.001
#    CUT,REN,OFF,TOL=a*10,1,0.015,0.001

    G,X,M,Z,R,A = np.array([0,0,0]),np.array([0,-0.5,0]),np.array([0.5,-0.5,0]),np.array([0,0,0.5]),np.array([0.5,0,0.5]),np.array([0.5,-0.5,0.5])

#    pt1,pt2,pt3 = np.zeros(3),np.array([0.2119,0.2119,0]),np.array([0.2119,-0.2119,0])

    
    spin_dict = {'bool':True,
        'soc':True,
        'lam':{0:0.00},
        'order':'N',
        'dS':0.0,
        'p_up':Fe1,
        'p_dn':Fe2}



    Bd = {'atoms':[0,0],
    			'Z':{0:26},
    			'orbs':[["32xy","32XY","32xz","32yz","32ZR"],["32xy","32XY","32xz","32yz","32ZR"]],
    			'pos':[Fe1,Fe2],
            'spin':spin_dict}
    
    Kd = {'type':'F',
              'avec':avec,
              'pts':[-0.2*M,G],#,Z,R,A,Z],
    			'grain':200,
    			'labels':['$\Gamma$','X','$M_y$','$\Gamma$','$M_x$']}
    
    
    Hd = {'type':'txt',
    			'filename':filenm,
    			'cutoff':CUT,
    			'renorm':REN,
    			'offset':OFF,
    			'tol':TOL,
    			'spin':spin_dict,
             'avec':avec}
    
    slab_dict = {'avec':avec,
          'miller':np.array([0,0,1]),
          'fine':(0,0),
          'thick':30,
          'vac':30,
          'termination':(0,0)}



 

    xmin,xmax,Nx=-0.25,0.25,131
    dx = (xmax-xmin)/Nx
    Imaps = np.zeros((20,Nx,150))
    Imaps_g = np.zeros((20,Nx,150))
    dS = 0.0
    OFF = 0
    Emax = np.zeros(20)
    dG = 0.029#0.03895


    so_vals = np.linspace(0,0.07,40)
    Bd = build_lib.gen_basis(Bd)
    Kobj = build_lib.gen_K(Kd)
    Hd = {'type':'txt',
			'filename':filenm,
			'cutoff':CUT,
			'renorm':REN,
			'offset':OFF,
			'tol':TOL,
			'spin':spin_dict,
         'avec':avec}
    
    TB_raw = build_lib.gen_TB(Bd,Hd,Kobj)
    TB = TB_raw.copy()
    
    vals = []
    oo_tol = 1e-4
    for so_i in range(len(so_vals)):
        OFF = 0.0
        dS = 0.02-0.0005*so_i
        spin_dict['lam'][0] = so_vals[so_i]
        Bd['bulk'] =redo_SO(TB.basis,spin_dict['lam'])
        Hd['offset']=0.0
        TB = build_lib.gen_TB(Bd,Hd,Kobj)
        TB.mat_els = redo_OO(TB.mat_els[:],dS)
#        
        not_aligned= True
        split_off = True
        count = 0
        while not_aligned or split_off:
        
            Hd['offset']=OFF
            TB = build_lib.gen_TB(Bd,Hd,Kobj)
            TB.mat_els = redo_OO(TB.mat_els[:],dS)
            TB.solve_H()
            if abs(TB.Eband[132][10])>0.00005:
                OFF += TB.Eband[132][10]
                TB.solve_H()
                not_aligned=True
            else:
#                print('satisfied k_F')
                not_aligned= False
            delta = TB.Eband[-1][8]+dG
            if delta>oo_tol:
#                dS +=delta/4
               dS +=oo_tol/2
               split_off = True
               
            elif delta<-oo_tol:
#                dS -=delta/4
                dS -=oo_tol/2
                split_off = True
            elif abs(delta)<=oo_tol:
                split_off = False
        vals.append([so_vals[so_i],OFF,dS])
        print(vals[-1])
#            print('offset: {:0.04f},shift by {:0.04f}'.format(TB.Eband[132][10],OFF),'band at Gamma: {:0.04f},change OO to {:0.04f}'.format(TB.Eband[-1][8],dS))
#            count+=1
#            if count<20:
#                if np.mod(count,2)==0:
#                    TB.plotting(-0.05,0.05)
                    
#        print(OFF,dS,TB.Eband[190][10],(TB.Eband[-1][10]-TB.Eband[-1][8])) 
#        Emax[so_i] = TB.Eband[-1][8]
    TB.solve_H()
    TB.plotting(-0.25,0.1)
    
    #    
    #    

#       
        
#        ARPES_dict={'cube':{'X':[xmin,xmax,Nx],'Y':[-4*dx,4*dx,9],'kz':0.0,'E':[-0.1,0.05,150]},
#                    'SE':[0.001],
#                    'directory':'C:\\Users\\rday\\Documents\\TB_ARPES\\2018\\TB_ARPES_2018\\FeSe',
#                    'hv': 37,
#                    'pol':np.array([0,-np.sqrt(0.5),np.sqrt(0.5)]),
#                    'mfp':7.0,
#                    'resolution':{'E':0.003,'k':0.005},
#                    'T':[True,100.0],
#                    'W':4.0,
#                    'angle':0,
#                    'spin':None,
#                    'slice':[False,-0.005]}
#                    #'Brads':{'0-3-2-1':100.0,'0-3-2-3':0.0}}
#    
#    #
#    #    
#    
#        expmt = ARPES.experiment(TB,ARPES_dict)
#        expmt.datacube(ARPES_dict)
#        expmt.plot_slice(ARPES_dict)
#        Itmp,Itmp_g = expmt.spectral(ARPES_dict,(1,4))
#        Imaps[so_i,:,:],Imaps_g[so_i,:,:] = Itmp[4,:,:],Itmp_g[4,:,:]
#        plt.title('FeSe SOC = {:0.04f} eV'.format(spin['lam'][0]))
##        plt.savefig('FeSe_SO_OO/ppol_{:d}_FeSe_SO_{:0.03f}eV_OO_{:0.03f}eV.png'.format(so_i,spin['lam'][0],dS))
        
#    fig = plt.figure()
#    kx = np.linspace(xmin,xmax,Nx)
#    En = np.linspace(-0.1,0.05,150)
#    pk_pts = np.array([np.where(abs(En-Emax[ii])==abs(En-Emax[ii]).min())[0][0] for ii in range(20)])
#
#    for plt_i in range(20):
#        plt.plot(kx,Imaps_g[plt_i,:,66])
#    plt.show()
#    plt.savefig('FeSe_SO_OO/ppol_MDCs_at_E_{:0.3f}.png'.format(En[66]))
#    fig2 = plt.figure()
#    for plt_i in range(20):
#        plt.plot(kx,Imaps_g[plt_i,:,pk_pts[plt_i]])
#        plt.savefig('FeSe_SO_OO/ppol_MDCs_at_Epk.png')
#        
    
###      
#    
##    expmt.plot_gui(ARPES_dict)
#
#
#    
#    
#	#####