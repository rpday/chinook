#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 12:24:04 2017

@author: ryanday
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys
sys.path.append('/Users/ryanday/Documents/UBC/TB_ARPES/TB_ARPES-master 4/')

import ubc_tbarpes.build_lib as build_lib
import ubc_tbarpes.ARPES_lib as ARPES

def SOC(L):
    
    HS = np.zeros((6,6),dtype=complex)
    HS[0,1]=-1.0j
    HS[0,5] = 1.0j
    HS[1,5]=-1
    HS[2,3]=-1.0j
    HS[2,4]=1.0
    HS[3,4]=1.0j
    HS +=np.conj(HS.T)
    HS*=L/2
    return HS


def FernVaf_remodel(kin,Ef,Exy,p_nem,L,Eg,txy,mG,b,cv):
    kmul = np.pi/0.83256
    k = kin*kmul
    #construct the low-energy effective model from the paper 2014 Nematicity and SOC in Pnictides
    pauli = np.array([[[1,0],[0,1]],[[0,1],[1,0]],[[0,-1.0j],[1.0j,0]],[[1,0],[0,-1]]])
    
    H = np.zeros((len(k),6,6),dtype=complex)
    H[:,:2,:2] = np.tensordot(Eg+mG*(k[:,0]**2+k[:,1]**2),pauli[0],0) + np.tensordot(b*(k[:,0]*k[:,1])+abs(p_nem),pauli[3],0) + cv*np.tensordot((k[:,0]**2-k[:,1]**2),pauli[1],0)
    H[:,3:5,3:5] = H[:,:2,:2]
    H[:,2,2] =  Exy + txy*(k[:,0]**2+k[:,1]**2)
    H[:,5,5] =  Exy + txy*(k[:,0]**2+k[:,1]**2)
    
    if abs(L)>0:
        H[:] += SOC(L)
    
    Eb,Ev = np.linalg.eigh(H)
    
    Eb+=Ef
  
    return Eb,Ev

        
    
  

def plot_dispersion(k,Eb):  
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if len(np.shape(k))>1:
        k = np.linalg.norm(k,axis=1)
    for i in range(6):
        ax.plot(k,Eb[:,i])
    ax.set_ylim(-0.25,0.05)
    ax.set_xlim(k.min(),k.max())
    plt.show()
    
        
  

def gen_fitfunc(Lval):
    txy = -0.02
    Eg=0.1
    mG =-0.29
    b=0.438
    cpar=0.174

    def fitfunc(kvalues,pa,pb,pc):
        disp,_ = FernVaf_remodel(kvalues,pb,pc,pa,L,Eg,txy,mG,b,cpar) 
        vals = np.array([disp[0,4],disp[1,0],disp[1,2]])
#        print(np.linalg.norm(vals-np.array([0.0,-0.049,-0.028])))
        return vals
    return fitfunc





def fit_model(p0,Lval):
    
    
    kv = np.array([[0.056,0.056],[0.0,0.0]])
    fitfunc = gen_fitfunc(Lval)
    ydata = np.array([0.0,-0.049,-0.028])
    fit_result,co = curve_fit(fitfunc,kv,ydata,sigma=(0.000001,0.000001,0.000001),absolute_sigma=True,p0=p0,bounds=(-0.7,0.7),max_nfev=20000)
    return fit_result




def gen_Hmat_function(args):
    Ef,Exy,p_nem,Lv,Eg,txy,mG,b,cv = args
    print(args)
    def Hfunc(kin):   
        
        kmul = np.pi/0.83256
        k = kin*kmul
        #construct the low-energy effective model from the paper 2014 Nematicity and SOC in Pnictides
        pauli = np.array([[[1,0],[0,1]],[[0,1],[1,0]],[[0,-1.0j],[1.0j,0]],[[1,0],[0,-1]]])
                
        H = np.zeros((len(k),6,6),dtype=complex)
        H[:,:2,:2] = np.tensordot(Eg+mG*(k[:,0]**2+k[:,1]**2),pauli[0],0) + np.tensordot(b*(k[:,0]*k[:,1])+abs(p_nem),pauli[3],0) + cv*np.tensordot((k[:,0]**2-k[:,1]**2),pauli[1],0)
        H[:,3:5,3:5] = H[:,:2,:2]
        H[:,2,2] =  Exy + txy*(k[:,0]**2+k[:,1]**2)
        H[:,5,5] =  Exy + txy*(k[:,0]**2+k[:,1]**2)
        if abs(Lv)>0:
            H[:] += SOC(Lv)
        H += Ef*np.identity(6)
        return H
    return Hfunc



    
#
def build_TB(args):
    
    a=3.7734 #FeSe
    c = 5.5258
    avec = np.array([[a,0,0.0],[0,a,0.0],[0.0,0.0,c]])

    G,X,M = np.array([0,0,0]),np.array([0.5,0.0,0]),np.array([0.5,0.5,0])
    spin = {'bool':True,'soc':False,'lam':{0:0}}
    
    Bd = {'atoms':[0],
          'Z':{0:26},
          'orbs':[['32xz','32yz','32xy']],
          'pos':[np.zeros(3)],
          'spin':spin,'slab':None}
    

    Kd = {'type':'F',
              'avec':avec,
			'pts':[0.5*M,G,0.5*X],
			'grain':200,
			'labels':['0.5 M','$\Gamma$','$0.5 X$']}
    

    Ham_dict = {'type':'list',
			'list':[],
			'cutoff':10,
			'renorm':1.0,
			'offset':0.0,
            'avec':avec,
			'tol':0.00001,
			'spin':spin}
    
    Bd = build_lib.gen_basis(Bd)
    Kobj = build_lib.gen_K(Kd)
    TB = build_lib.gen_TB(Bd,Ham_dict,Kobj)
    TB.mat_els = []
    TB.alt_ham = gen_Hmat_function(args)
    return TB
    
    


if __name__=="__main__":
    

    lenk = 400
    klim = 0.5*np.sqrt(0.5)
    kpts = np.array([[-klim+klim*i/lenk,-klim+klim*i/lenk] for i in range(lenk)])#+[[klim*i/lenk,0] for i in range(lenk)])
    kpts = np.array([[0.056,0.056],[0,0]])

#    Ef = -0.1
#    Exy = 0.05
    
    txy = -0.02
    Eg=0.1
    mG =-0.29
    b=0.438
    cpar=0.174
    L = 0.02
#    p_nem=0.0
#    Eb,Ev = FernVaf_remodel(kpts,Ef,Exy,p_nem,L,Eg,txy,mG,b,cpar)
#    Eb,Ev = FernVaf_remodel(k,ax,ay,b,c,E,Exy,txy,p,L)
#    plot_dispersion(kpts,Eb)
    
#    TB = build_TB()
    
    
    p0 = [0.0,-0.1,0.05]
    Nfits = 30
    cfits = np.zeros((Nfits,3))
    Lvals = np.linspace(0,0.05,Nfits)
    cfit = fit_model(p0,L)
#    for ii in range(Nfits):
#        L = Lvals[ii]
#        cfits[ii] = fit_model(p0,L)
#        p0 = cfits[ii]
#    
#    k = np.array([[-klim+klim*i/lenk,-klim+klim*i/lenk] for i in range(lenk)])#+[[klim*i/lenk,0] for i in range(lenk)])
#    for ii in range(Nfits):
#        Eb,Ev = FernVaf_remodel(k,cfits[ii,1],cfits[ii,2],cfits[ii,0],Lvals[ii],Eg,txy,mG,b,cpar)
#        plot_dispersion(k,Eb)
        
        

####        
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    ax.plot(Lvals,cfits[:,0])
#    ax.plot(Lvals,cfits[:,1])
#    ax.plot(Lvals,cfits[:,2])
#    ax.legend(['OO','Eoff','Edxy'])
    
    
    args = [cfit[1],cfit[2],cfit[0],L,Eg,txy,mG,b,cpar]
#    args = [-0.1,0.05,0.0,0.0,Eg,txy,mG,b,cpar]
    TB = build_TB(args)
    _ = TB.solve_H()
    TB.plotting()
    
    
    ARPES_dict={'cube':{'X':[-0.35,0.35,60],'Y':[-0.35,0.35,60],'kz':0.0,'E':[-0.25,0.05,300]},
        'SE':[0.01,0.0,0.4],
        'directory':'/Users/ryanday/Documents/UBC/TB_ARPES-082018/examples/FeSe',
        'hv': 37,
        'pol':np.array([1,0,0]),
        'mfp':7.0,
        'slab':False,
        'resolution':{'E':0.065,'k':0.01},
        'T':[True,10.0],
        'W':4.0,
        'angle':np.pi/4,
        'spin':None,
        'slice':[False,-0.005]}
    
    expmt = ARPES.experiment(TB,ARPES_dict)
    expmt.plot_gui(ARPES_dict)
##    
    
    
    
    
#