# -*- coding: utf-8 -*-
"""
Created on Wed Feb 07 12:07:08 2018

@author: rday
"""


    
import numpy as np
import matplotlib.pyplot as plt
import build_lib
import ARPES_lib as ARPES
import scipy.ndimage as nd
import operator_library as ops
import plt_sph_harm as sph
import matplotlib.cm as cm

if __name__=="__main__":
#def run_calc(filename):
    a,c =  3.7914,6.3639
    avec = np.array([[a/np.sqrt(2),a/np.sqrt(2),0.0],[-a/np.sqrt(2),a/np.sqrt(2),0.0],[0.0,0.0,c]])

    filenm ="LiFeAs_test2.txt"
    CUT,REN,OFF,TOL=a*3,1/2.17,0.030,0.001
    G,X,M,Z,A = np.array([0,0,0]),np.array([0,0.5,0]),np.array([0.5,0.5,0]),np.array([0,0,0.5]),np.array([0.5,0.5,0.5])

	

    
    slab_dict = {'bool':False,
                'hkl':np.array([0,0,1]),
                'cells':20,
                'buff':8,
                'term':0,
                'avec':avec}

    basis_dict = {'atoms':[0,0],
			'Z':{0:26},
			'orbs':[["32xy","32XY","32xz","32yz","32ZR"],["32xy","32XY","32xz","32yz","32ZR"]],
			'pos':[np.array([-a/np.sqrt(8),0,0]),np.array([a/np.sqrt(8),0,0])],
            'slab':slab_dict}

    K_dict = {'type':'F',
			'pts':[0.3*X+0.1*Z,0.1*Z,0.3*X+0.1*Z],#
			'grain':500,
			'labels':['$0.3X$','$\Gamma$','$0.3X$','$\Gamma$','$Z$']}
    
    spin_dict = {'soc':True,'lam':{0:0.04/2.17}}

    Ham_dict = {'type':'txt',
			'filename':filenm,
			'cutoff':CUT,
			'renorm':REN,
			'offset':OFF,
			'tol':TOL,
			'so':spin_dict['soc']}
    
        

 
    	#####
    basis_dict = build_lib.gen_basis(basis_dict,spin_dict)
    Kobj = build_lib.gen_K(K_dict,avec)
    TB = build_lib.gen_TB(basis_dict,Ham_dict,Kobj)
    
#    for ii in range(len(TB.mat_els)):
#        if TB.mat_els[ii].i==TB.mat_els[ii].j:
#            if ii<len(TB.basis)/2:
#                TB.mat_els[ii].H.append([0,0,0,1.0e-9])
#            else:
#                TB.mat_els[ii].H.append([0,0,0,-1.0e-9])
            
    
    
    TB.solve_H()
    TB.plotting(-0.25,0.1)
#    OS = ops.Sz(TB,vlims=(-0.5,0.5),Elims=(-0.25,0.1))
#    OL = ops.Lz_path(TB,vlims=(-1,1),Elims=(-0.25,0.1))
#    OLZ = ops.LzSz_path(TB,vlims=(-0.5,0.5),Elims=(-0.25,0.1))
    O = ops.LdotS(TB,axis='z',vlims=(-0.5,0.5),Elims=(-0.25,0.05))
###    OJ = ops.Jz_path(TB,vlims=(-0.5,0.5),Elims=(-0.25,0.1))
#    ktuple = ((-0.4,0.4,800),(-0.4+1.17183176,0.4+1.17183176,800),0.0)
#    Ef = -0.017
#    tol=0.0008
#    LDS = ops.LSmat(TB)
#    FSLS = ops.O_surf(LDS,TB,ktuple,Ef,tol,vlims=(-0.1,0.1))
#    FS = ops.FS(TB,ktuple,Ef,tol)
#    d1d,d1u = {o.index:o.proj for o in TB.basis[:5]},{(o.index-5):o.proj for o in TB.basis[10:15]}
#    d2d,d2u = {(o.index-5):o.proj for o in TB.basis[5:10]},{(o.index-10):o.proj for o in TB.basis[15:]}
#    d1 = {**d1d,**d1u}
#    d2 = {**d2d,**d2u}
#    ind = 8
#    psi_11 = np.array([list(TB.Evec[i,:5,ind])+list(TB.Evec[i,10:15,ind]) for i in range(len(Kobj.kpts))])
#
#    psi_12 = np.array([list(TB.Evec[i,5:10,ind])+list(TB.Evec[i,15:,ind]) for i in range(len(Kobj.kpts))])
#   
#    psi_1 = np.array([list(TB.Evec[i,:,ind+2]) for i in range(len(Kobj.kpts))])
#    df = {o.index:o.proj for o in TB.basis}
#    psi_11 = np.array([list(TB.Evec[i,:5,ind:ind+2])+list(TB.Evec[i,10:15,ind:ind+2]) for i in range(len(Kobj.kpts))])
#    
#    for ki in range(len(Kobj.kpts)):
#        psi = psi_11[ki]
#        strnm = 'psi_plots\\03_05\\so_GZ1_1_{:d}.png'.format(ki)
#        sph.gen_psi(30,psi_11[ki],d1,strnm)
#####        
#####    ki = 6
##    psi = psi_21[ki,:]
##    strnm = 'psi_plots\\no_so_GZ2_1_{:d}.png'.format(ki)
#    sph.gen_psi(20,psi_11[ki],d1)
#    sph.gen_psi(20,psi_21[ki],d1)
#
#    sph.gen_psi(20,psi_31[ki],d1)
#
#    sph.gen_psi(20,psi_41[ki],d1)

#    sph.gen_psi(20,psi_12[ki],d1)
#    sph.gen_psi(20,psi_22[ki],d1)
#
#    sph.gen_psi(20,psi_32[ki],d1)
#
#    sph.gen_psi(20,psi_42[ki],d1)
    
##    
##    
##    
#    ARPES_dict={'cube':{'X':[-0.4,0.4,200],'Y':[-0.4,0.4,200],'kz':0.1*np.pi/c,'E':[-0.45,0.3,400]},
#                'SE':[0.005,0,0],
#                'directory':'C:\\Users\\rday\\Documents\\TB_ARPES\\2018\\TB_ARPES_2018\\LiFeAs',
#                'hv': 26.0,
#                'pol':np.array([0,1,0]),
#                'mfp':7.0,
#                'resolution':{'E':0.07,'k':0.05},
#                'T':[True,20.0],
#                'W':4.0,
#                'angle':np.pi/4,
#                'spin':None,
#                'slice':[False,-0.1]}
#                #'mirror':np.array([0,1,1])}
#
##
##    
##
#    exp = ARPES.experiment(TB,ARPES_dict)
#    exp.datacube(ARPES_dict)
#    
#    ARPES_dict['pol'] = np.array([0.5,-np.sqrt(0.5)*1.0j,np.sqrt(0.5)*1.0j])
#    ARPES_dict['spin'] = [1,np.array([0,0,1])]
#    CMU,CMUg = exp.spectral(ARPES_dict)
#    ARPES_dict['spin'] = [-1,np.array([0,0,1])]
#    CMD,CMDg = exp.spectral(ARPES_dict)
#    ARPES_dict['pol'] = np.array([0.5,np.sqrt(0.5)*1.0j,-np.sqrt(0.5)*1.0j])
#    ARPES_dict['spin'] = [1,np.array([0,0,1])]
#    CPU,CPUg = exp.spectral(ARPES_dict)
#    ARPES_dict['spin'] = [-1,np.array([0,0,1])]
#    CPD,CPDg = exp.spectral(ARPES_dict)
##    
###    
##    CPSg  = (np.sqrt(CPUg*CMDg)-np.sqrt(CPDg*CMUg))/(np.sqrt(CPUg*CMDg)+np.sqrt(CPDg*CMUg)+0.0001)
##    CPS = (np.sqrt(CPU*CMD)-np.sqrt(CPD*CMU))/(np.sqrt(CPUg*CMDg)+np.sqrt(CPD*CMU)+0.0001)
#    IP = np.sqrt((CPU+0.00001)*(CMD+0.00001))
#    IA = np.sqrt((CPD+0.00001)*(CMU+0.00001))
#    IPg = np.sqrt((CPUg+0.00001)*(CMDg+0.00001))
#    IAg = np.sqrt((CPDg+0.00001)*(CMUg+0.00001))
#    
#    CPSg = (IPg-IAg)/(IPg+IAg)
#    CPS = (IP-IA)/(IP+IA)
#
#    x,e = np.linspace(*ARPES_dict['cube']['X']),np.linspace(*ARPES_dict['cube']['E'])
#    X,E = np.meshgrid(x,e)
#
##    
##    
#
##    
##
##    
##
###    Ig=exp.plot_gui(ARPES_dict) #If you want to see another polarization, simply update ARPES_dict['pol'] and re-run exp.plot_slice(ARPES_dict)
##
##
#    fig = plt.figure()
#    ax = fig.add_subplot(131)
#    ax2 = fig.add_subplot(132)
#    ax3 = fig.add_subplot(133)
#    p1 = ax.pcolormesh(X,E,IP[100,:,:].T,cmap=cm.Greys_r)
#    p2 = ax2.pcolormesh(X,E,IA[100,:,:].T,cmap=cm.Greys_r)
#    p3 = ax3.pcolormesh(X,E,CPS[100,:,:].T,cmap=cm.RdBu,vmin= -0.3,vmax=0.3)
#    plt.savefig('CPS_LiFeAs.pdf',transparent=True,rasterized=True)
#
###
##    
#    return TB,Kobj
###	#####
###    
#if __name__=="__main__":
#    TB,K=run_calc("LiFeAs_test2.txt")