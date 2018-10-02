# -*- coding: utf-8 -*-
"""
Created on Wed Feb 07 12:07:08 2018

@author: rday
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
import sys
sys.path.append('C:\\Users\\rday\\Documents\\TB_ARPES\\2018\\TB_ARPES_2018')

    
import numpy as np
import matplotlib.pyplot as plt
import ubc_tbarpes.build_lib as build_lib
import ubc_tbarpes.ARPES_lib as ARPES
import scipy.ndimage as nd
import ubc_tbarpes.operator_library as ops
import ubc_tbarpes.plt_sph_harm as sph
import matplotlib.cm as cm

if __name__=="__main__":
#def run_calc(filename):
    a,c =  3.7914,6.3639
    avec = np.array([[a/np.sqrt(2),a/np.sqrt(2),0.0],[-a/np.sqrt(2),a/np.sqrt(2),0.0],[0.0,0.0,c]])

    filenm ="C:\\Users\\rday\\Documents\\TB_ARPES\\2018\\TB_ARPES_2018\\Materials\\LiFeAs_test4.txt"
    CUT,REN,OFF,TOL=a*3,1/2.17,0.030,0.001
    G,Y,X,M,Z,A = np.array([0,0,0]),np.array([0,0.5,0]),np.array([0.5,0,0]),np.array([0.5,0.5,0]),np.array([0,0,0.5]),np.array([0.5,0.5,0.5])

	

    
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
			'pts':[X,G,M,G,Z],
			'grain':2000,
			'labels':['$X$','$\Gamma$','$M$','$\Gamma$','$Z$']}
    
    spin_dict = {'soc':False,'lam':{0:0.04/2.17}}

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
##            
#    
#    
    TB.solve_H()
    TB.plotting(-0.25,0.1)
#    OS = ops.Sz(TB,vlims=(-0.5,0.5),Elims=(-0.25,0.1))
#    OL = ops.Lz_path(TB,vlims=(-1,1),Elims=(-0.25,0.1))
#    OLZ = ops.LzSz_path(TB,vlims=(-0.5,0.5),Elims=(-0.25,0.1))
#    O = ops.LdotS(TB,axis='z',vlims=(-0.5,0.5),Elims=(-0.1,0.02))
####    OJ = ops.Jz_path(TB,vlims=(-0.5,0.5),Elims=(-0.25,0.1))
#    ktuple = ((-0.1,1.2,800),(-0.1,1.2,800),0.0)
#    Ef = -0.020
#    tol=0.0008
###    LDS = ops.LSmat(TB)
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
##    psi_1 = np.array([list(TB.Evec[i,:,ind+2]) for i in range(len(Kobj.kpts))])
##    df = {o.index:o.proj for o in TB.basis}
##    psi_11 = np.array([list(TB.Evec[i,:5,ind:ind+2])+list(TB.Evec[i,10:15,ind:ind+2]) for i in range(len(Kobj.kpts))])
##    
#    for ki in range(len(Kobj.kpts)):
#        strnm = 'psi_plots\\03_05\\so_GZ1_1_{:d}.png'.format(ki)
#        print('doing it')
#        sph.gen_psi(30,psi_11[ki],d1,strnm)
####        
####    ki = 6
#    psi = psi_21[ki,:]
#    strnm = 'psi_plots\\no_so_GZ2_1_{:d}.png'.format(ki)
#    sph.gen_psi(20,psi_11[ki],d1)
#    sph.gen_psi(20,psi_21[ki],d1)
#
#    sph.gen_psi(20,psi_31[ki],d1)
#
#    sph.gen_psi(20,psi_41[ki],d1)
#
#    sph.gen_psi(20,psi_12[ki],d1)
#    sph.gen_psi(20,psi_22[ki],d1)
#
#    sph.gen_psi(20,psi_32[ki],d1)
#
#    sph.gen_psi(20,psi_42[ki],d1)
    
##    
##    
##    
#    ARPES_dict={'cube':{'X':[-0.6,0.6,60],'Y':[-0.6,0.6,60],'kz':0.1*np.pi/c,'E':[-0.45,0.3,300]},
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
#
#    expmt = ARPES.experiment(TB,ARPES_dict)
#    expmt.datacube(ARPES_dict)
#    Ig=expmt.plot_gui(ARPES_dict) #If you want to see another polarization, simply update ARPES_dict['pol'] and re-run exp.plot_slice(ARPES_dict)

#    
#    ARPES_dict['pol'] = np.array([0.5,-np.sqrt(0.5)*1.0j,np.sqrt(0.5)*1.0j])
#    ARPES_dict['spin'] = [1,np.array([1,0,0])]
#    CMU,CMUg = exp.spectral(ARPES_dict)
#    ARPES_dict['spin'] = [-1,np.array([1,0,0])]
#    CMD,CMDg = exp.spectral(ARPES_dict)
#    ARPES_dict['pol'] = np.array([0.5,np.sqrt(0.5)*1.0j,-np.sqrt(0.5)*1.0j])
#    ARPES_dict['spin'] = [1,np.array([1,0,0])]
#    CPU,CPUg = exp.spectral(ARPES_dict)
#    ARPES_dict['spin'] = [-1,np.array([1,0,0])]
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
##
#    x,e = np.linspace(*ARPES_dict['cube']['X']),np.linspace(*ARPES_dict['cube']['E'])
#    X,E = np.meshgrid(x,e)
##
##    
##    
#
##    
##
##    
##
##
###
#    fig = plt.figure()
#    ax = fig.add_subplot(131)
#    ax2 = fig.add_subplot(132)
#    ax3 = fig.add_subplot(133)
#    p1 = ax.pcolormesh(X,E,IP[20,:,:].T,cmap=cm.Greys_r)
#    p2 = ax2.pcolormesh(X,E,IA[20,:,:].T,cmap=cm.Greys_r)
#    p3 = ax3.pcolormesh(X,E,CPS[20,:,:].T,cmap=cm.RdBu,vmin= -0.3,vmax=0.3)
##    plt.savefig('CPS_LiFeAs.pdf',transparent=True,rasterized=True)
##
####
###    
#    return TB,Kobj
###	#####
###    
#if __name__=="__main__":
#    TB,K=run_calc("LiFeAs_test2.txt")