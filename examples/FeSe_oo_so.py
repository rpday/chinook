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
import scipy.ndimage as nd
import ubc_tbarpes.operator_library as ops
import ubc_tbarpes.plt_sph_harm as sph
import ubc_tbarpes.direct as direct
import ubc_tbarpes.direct as integral

if __name__=="__main__":
    a,c =  3.7734,5.5258 
    avec = np.array([[a/np.sqrt(2),a/np.sqrt(2),0.0],[-a/np.sqrt(2),a/np.sqrt(2),0.0],[0.0,0.0,c]])

    filenm = 'FeSe_BMA_mod.txt'
#    CUT,REN,OFF,TOL=a*3,1/1.4,0.12,0.001
#    CUT,REN,OFF,TOL=a*10,1,0.015,0.001

    G,X,M,Z,R,A = np.array([0,0,0]),np.array([0,-0.5,0]),np.array([0.5,-0.5,0]),np.array([0,0,0.5]),np.array([0.5,0,0.5]),np.array([0.5,-0.5,0.5])

#    pt1,pt2,pt3 = np.zeros(3),np.array([0.2119,0.2119,0]),np.array([0.2119,-0.2119,0])

    
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
			'pts':[-M,G],
			'grain':200,
			'labels':['M','$\Gamma$','M','X','$\Gamma$']}



 
    	#####
    spin = {'soc':True,'lam':{0:0.005}}

    xmin,xmax,Nx=-0.25,0.25,131
    dx = (xmax-xmin)/Nx
    Imaps = np.zeros((20,Nx,150))
    Imaps_g = np.zeros((20,Nx,150))
    dS = 0.019
    OFF = 0.0
    Emax = np.zeros(20)
    dG = 0.029#0.03895


    so_vals = np.linspace(0,0.03,20)
    for so_i in range(1):
        spin['lam'][0] = so_vals[-1]
        print(spin)
        not_aligned= True
        split_off = True
        while not_aligned or split_off:
            CUT,REN,TOL=a*10,1,0.001
            
            Hd = {'type':'txt',
        			'filename':filenm,
        			'cutoff':CUT,
        			'renorm':REN,
        			'offset':OFF,
        			'tol':TOL,
        			'so':spin['soc']}
            Bd = build_lib.gen_basis(Bd,spin)
            Kobj = build_lib.gen_K(Kd,avec)
            TB = build_lib.gen_TB(Bd,Hd,Kobj)
        
        #    dB = -0.0089

            for ii in range(len(TB.mat_els)):
                if TB.mat_els[ii].i==TB.mat_els[ii].j:
                    if TB.mat_els[ii].i%5==2:
                        TB.mat_els[ii].H.append([0,0,0,-abs(dS)])
                    elif TB.mat_els[ii].i%5==3:
                        TB.mat_els[ii].H.append([0,0,0,abs(dS)])
        ##                
        
            TB.solve_H()
            if abs(TB.Eband[190][10])>0.001:
                OFF += TB.Eband[190][10]
                not_aligned=True
            else:
                not_aligned= False
            if (TB.Eband[-1][10]-TB.Eband[-1][8]-dG)>0.00001:
               dS -=0.00001
               split_off = True
            elif (TB.Eband[-1][10]-TB.Eband[-1][8]-dG)<-0.00001:
                dS +=0.00001
                split_off = True
            elif abs(TB.Eband[-1][10]-TB.Eband[-1][8]-dG)<0.00001:
                split_off = False
            
        print(OFF,dS,TB.Eband[190][10],(TB.Eband[-1][10]-TB.Eband[-1][8])) 
        Emax[so_i] = TB.Eband[-1][8]
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