#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 12:17:01 2017

@author: ryanday
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
sys.path.append('C:/Users/rday/Documents/TB_ARPES/2018/TB_ARPES_2018/TB_ARPES-master/')
import matplotlib as mpl
import ubc_tbarpes.build_lib as build_lib
import ubc_tbarpes.ARPES_lib as ARPES
import ubc_tbarpes.optics as optics
import ubc_tbarpes.operator_library as ops
import ubc_tbarpes.dos_Tk as dos_TK
import ubc_tbarpes.dos as dos
import Kreisel_load as Kreisel



def redo_SO(basis,lamdict):
    for bi in basis:
        bi.lam = lamdict[bi.atom]
    
    return basis

def orbital_order(mat_els,d):
    for hi in mat_els:
        if np.mod(hi.i,5)==2 and hi.i==hi.j:
            hi.H.append([0,0,0,-d])
        elif np.mod(hi.i,5)==3 and hi.i==hi.j:
            hi.H.append([0,0,0,d])
    return mat_els
            


#if __name__=="__main__":
    
    
############################ STRUCTURAL PARAMETERS ############################
a=3.7734 #FeSe
c = 5.5258
avec = np.array([[a/np.sqrt(2),a/np.sqrt(2),0.0],[-a/np.sqrt(2),a/np.sqrt(2),0.0],[0.0,0.0,c]])
Fe1,Fe2 = np.array([-a/np.sqrt(8),0,0]),np.array([a/np.sqrt(8),0,0])


#Kreisel Parameters:
#a,b,c = 2.665,2.655,5.48
#a,b,c = 2.66,2.66,5.48
#avec = np.array([[a,b,0.0],[a,-b,0.0],[0.0,0.0,c]])
#Fe1,Fe2 = np.array([0,0,0]),np.array([a,0,0])
G,X,M,Z,R,A = np.array([0,0,0]),np.array([0.5,0.0,0]),np.array([0.5,0.5,0]),np.array([0,0,0.5]),np.array([0.5,0,0.5]),np.array([0.5,-0.5,0.5])
Mx = np.array([0.5,-0.5,0.0])

############################ HAMILTONIAN PARAMETERS ############################

#filenm ='FeSe_BMA_BO.txt'
#CUT,REN,OFF,TOL=a*3,1/1.,0.01,1e-7
filenm = 'FeSe_o.txt'
CUT,REN,OFF,TOL=a*3,1/1.4,0.12,0.001

#Hlist = Kreisel.gen_Kreisel_list(avec,[Fe1,Fe2])
#filenm = 'FeSe_Kreisel_mod.txt'
#filenm = 'FeSe_BMA_MOD.txt'
#CUT,REN,TOL=a*4,1.0,1e-7
#OFF = 0.015

######################### MODEL GENERATION PARAMETERS ##########################

spin_dict = {'bool':False,
        'soc':True,
        'lam':{0:0.04},
        'order':'N',
        'dS':0.0,
        'p_up':Fe1,
        'p_dn':Fe2}



basis_dict = {'atoms':[0,0],
			'Z':{0:26},
			'orbs':[["32xy","32XY","32xz","32yz","32ZR"],["32xy","32XY","32xz","32yz","32ZR"]],
			'pos':[Fe1,Fe2],
        'spin':spin_dict}

K_dict = {'type':'F',
          'avec':avec,
          'pts':[M,G,Z],#,Z,R,A,Z],
#			'pts':[np.array([-0.72044,0,0]),np.array([1.0165866,0,0])],
			'grain':200,
			'labels':['$\Gamma$','X','$M_y$','$\Gamma$','$M_x$']}


ham_dict = {'type':'txt',
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

######################### OPTICS EXPERIMENT PARAMETERS #########################


optics_dict = {'hv':0.36,
      'linewidth':0.01,
      'lifetime':0.02,
      'mesh':40,
      'T':10,
      'pol':np.array([1,0,0])}

######################### ARPES EXPERIMENT PARAMETERS #########################


ARPES_dict={'cube':{'X':[-0.5,0.5,20],'Y':[-0.5,0.5,20],'kz':0.0,'E':[-0.2,0.05,100]},
        'SE':[0.01,0.0,0.4],
        'directory':'/Users/ryanday/Documents/UBC/TB_ARPES-082018/examples/FeSe',
        'hv': 37,
        'pol':np.array([1,0,0]),
        'mfp':7.0,
        'slab':True,
        'resolution':{'E':0.065,'k':0.01},
        'T':[True,10.0],
        'W':4.0,
        'angle':0.0,
        'spin':None,
        'slice':[False,-0.005],
        'Brads':{'0-3-2-1':1.0,'0-3-2-3':1.0}}
 
################################# BUILD MODEL #################################

def build_TB():
    BD = build_lib.gen_basis(basis_dict)
    Kobj = build_lib.gen_K(K_dict)
    TB = build_lib.gen_TB(BD,ham_dict,Kobj)
    return TB


def ARPES_run(basis_dict,ham_dict,ARPES_dict,vfile):
    pars = []
    with open(vfile,'r') as fromfile:
        for line in fromfile:
            pars.append([float(ti) for ti in line.split(',')])
    fromfile.close()
#    pmax = 5
    pars = np.array(pars)
    basis_dict = build_lib.gen_basis(basis_dict)
    Imaps = np.zeros((len(pars),ARPES_dict['cube']['X'][2],ARPES_dict['cube']['E'][2]))
    Imapp = np.zeros((len(pars),ARPES_dict['cube']['X'][2],ARPES_dict['cube']['E'][2]))
    for p in list(enumerate(pars)):
        ham_dict['offset']=p[1][1]
        spin_dict['lam'][0] = p[1][0]
        basis_dict['bulk'] = redo_SO(basis_dict['bulk'],spin_dict['lam'])
        TB = build_lib.gen_TB(basis_dict,ham_dict)
        TB.mat_els = orbital_order(TB.mat_els[:],p[1][2])
        ARPES_expmt = ARPES.experiment(TB,ARPES_dict)
        ARPES_expmt.datacube(ARPES_dict)
        ARPES_dict['pol']=np.array([1,0,0])
        I,Ig = ARPES_expmt.spectral(ARPES_dict)
        Imaps[p[0]] = I[0,:,:]
        ARPES_dict['pol']=np.array([0,0.707,0.707])
        I,Ig = ARPES_expmt.spectral(ARPES_dict)
        Imapp[p[0]] = I[0,:,:]
    return pars,Imapp,Imaps
        


if __name__ == "__main__":
    TB = build_TB()
#    TB.mat_els = orbital_order(TB.mat_els,0.01)
#    TB.solve_H()
    TB.plotting()
    
#    pars,Ip,Is = ARPES_run(basis_dict,ham_dict,ARPES_dict,'SOC_OFF_OO.txt')
##    
#    fig = plt.figure()
#    ax = fig.add_subplot(121)
#    ax2 = fig.add_subplot(122)
#    x = np.linspace(*ARPES_dict['cube']['X'])
#    s = pars[:,0]
#    X,S = np.meshgrid(x,s)
#    ax.pcolormesh(X,S,Ip[:,:,68],cmap=cm.magma)
#    ax2.pcolormesh(X,S,Is[:,:,68],cmap=cm.magma)
#    Svx = ops.S_vec(len(TB.basis),np.array([1,0,0]))
#    Svy = ops.S_vec(len(TB.basis),np.array([0,1,0]))
#    Dsurf = np.identity(len(TB.basis))*np.array([np.exp(bi.depth/10) for bi in TB.basis])
#    Sxs = np.dot(Svx,Dsurf)
#    Sys = np.dot(Svy,Dsurf)
#    SX = ops.O_path(Sxs,TB,Kobj=TB.Kobj,vlims=(-0.25,0.25),Elims=(-1.5,1.5),degen=True)
#    LS = ops.LdotS(TB,None)
#    SY = ops.O_path(Sys,TB,Kobj=TB.Kobj,vlims=(-0.25,0.25),Elims=(-1.5,1.5),degen=True)
#    DO = dos_TK.dos_interface(TB,0.005)
##    DO = dos.dos_env(TB)
#    DO.do_dos((20,20,20))
    
############################## CHARACTERIZE MODEL ##############################

                
#    TB.solve_H()
#    TB.plotting(-0.45,0.15)
#    sproj = oper.O_path(oper.surface_projection(TB,10),TB,vlims=(0,1),Elims=(-0.5,0.2))
#    
################################# EXPERIMENTS #################################

    
#    optics_exp = optics.optical_experiment(TB,optics_dict)
#    optics_exp.integrate_jdos()
#
#    ARPES_expmt = ARPES.experiment(TB,ARPES_dict)
#    ARPES_expmt.datacube(ARPES_dict)
##    ARPES_expmt.plot_gui(ARPES_dict)
#    Ip,Ipg = ARPES_expmt.spectral(ARPES_dict)
##    ARPES_dict['spin']=[1,np.array([1,0,0])]
##    Isu,Isug = ARPES_expmt.spectral(ARPES_dict)
##    ARPES_dict['spin'] = [-1,np.array([1,0,0])]
##    Isd,Isdg = ARPES_expmt.spectral(ARPES_dict)
#    
#    
#    
#    
#    x = np.linspace(*ARPES_dict['cube']['X'])
#    y = np.linspace(*ARPES_dict['cube']['Y'])
#    w = np.linspace(*ARPES_dict['cube']['E'])
#    mpl.rcParams['font.size']=14
#    fig = plt.figure()
#    ax1 = fig.add_subplot(131)
#    ax2 = fig.add_subplot(132)
#    ax3 = fig.add_subplot(133)
#    X1,Y1 = np.meshgrid(x,y)
#    X2,Y2 = np.meshgrid(x,w)
#    X3,Y3 = np.meshgrid(y,w)
#    wind = np.where(abs(w)==abs(w).min())[0][0]
#    ax1.pcolormesh(X1,Y1,Ipg[:,:,wind],cmap=cm.magma)
#    ax2.pcolormesh(X3,Y3,Ipg[:,5,:].T,cmap=cm.magma)
#    ax3.pcolormesh(X2,Y2,Ipg[5,:,:].T,cmap=cm.magma)
#    ax1.set_xlabel('Momentum x (1/A)')
#    ax1.set_ylabel('Momentum y (1/A)')
#    ax2.set_xlabel('Momentum y (1/A)')
#    ax2.set_ylabel('Energy (eV)')
#    ax3.set_xlabel('Momentum x (1/A)')
#    ax1.set_aspect(1)
#    ax2.set_aspect(6)
#    ax3.set_aspect(6)