#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 12:24:04 2017

@author: ryanday
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm

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
#    H[:,:2,:2] = np.tensordot(Eg+mG*(k[:,0]**2+k[:,1]**2),pauli[0],0) + np.tensordot(b*(k[:,0]*k[:,1])+abs(p_nem),pauli[3],0) + cv*np.tensordot((k[:,0]**2-k[:,1]**2),pauli[1],0)
#    H[:,3:5,3:5] = H[:,:2,:2]
#    H[:,2,2] =  Exy + txy*(k[:,0]**2+k[:,1]**2)
#    H[:,5,5] =  Exy + txy*(k[:,0]**2+k[:,1]**2)
    
    H[:,:2,:2] = np.tensordot(Eg+mG*(k[:,0]**2+k[:,1]**2),pauli[0],0) + np.tensordot(b*0.5*(k[:,0]**2-k[:,1]**2)+abs(p_nem),pauli[3],0) + cv*np.tensordot((2*k[:,0]*k[:,1]),pauli[1],0)
    H[:,3:5,3:5] = H[:,:2,:2]
    H[:,2,2] =  Exy + txy*(k[:,0]**2+k[:,1]**2)
    H[:,5,5] =  Exy + txy*(k[:,0]**2+k[:,1]**2)
    
    H += np.identity(6)*Ef
    if abs(L)>0:
        H[:] += SOC(L)
    
    Eb,Ev = np.linalg.eigh(H)
    
  
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
#        H[:,:2,:2] = np.tensordot(Eg+mG*(k[:,0]**2+k[:,1]**2),pauli[0],0) + np.tensordot(b*(k[:,0]*k[:,1])+abs(p_nem),pauli[3],0) + cv*np.tensordot((k[:,0]**2-k[:,1]**2),pauli[1],0)
#        H[:,3:5,3:5] = H[:,:2,:2]
#        H[:,2,2] =  Exy + txy*(k[:,0]**2+k[:,1]**2)
#        H[:,5,5] =  Exy + txy*(k[:,0]**2+k[:,1]**2)
        
        H[:,:2,:2] = np.tensordot(Eg+mG*(k[:,0]**2+k[:,1]**2),pauli[0],0) + np.tensordot(b*0.5*(-k[:,0]**2+k[:,1]**2)+abs(p_nem),pauli[3],0) + cv*np.tensordot((2*k[:,0]*k[:,1]),pauli[1],0)
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
    avec = np.array([[a/np.sqrt(2),a/np.sqrt(2),0.0],[-a/np.sqrt(2),a/np.sqrt(2),0.0],[0.0,0.0,c]])

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
    
    mpl.rcParams['font.size'] = 12
    mpl.rcParams['axes.linewidth']=1.5
    mpl.rcParams['xtick.labelsize']=12
    mpl.rcParams['ytick.labelsize']=12
    

    lenk = 400
    klim = 0.5
    kpts = np.array([[-klim+klim*i/lenk,0] for i in range(lenk)])#+[[klim*i/lenk,0] for i in range(lenk)])
#    kpts = np.array([[0.056,0.056],[0,0]])

    Ef = -0.1
    Exy = 0.05
    
    txy = -0.02
    Eg=0.1
    mG =-0.29
    b=0.438
    cpar=0.224
    L = 0.04
    p_nem=0.0
    Eb,Ev = FernVaf_remodel(kpts,Ef,Exy,p_nem,L,Eg,txy,mG,b,cpar)
    plot_dispersion(kpts,Eb)
    
    
    ARPES_dict={'cube':{'X':[-0.0,0.0,1],'Y':[-0.5,0.5,140],'kz':0.0,'E':[-0.25,0.05,400]},
        'SE':[0.01,0.0,0.8],
        'directory':'/Users/ryanday/Documents/UBC/TB_ARPES-082018/examples/FeSe',
        'hv': 37,
        'pol':np.array([1,0,1]),
        'mfp':7.0,
        'slab':False,
        'resolution':{'E':0.03,'k':0.1},
        'T':[True,120.0],
        'W':4.0,
        'angle':0.0,
        'spin':None,
        'slice':[False,-0.005]}
    
#    TB = build_TB()
    
    
    p0 = [0.0,-0.1,0.05]
    Nfits = 60
    cfits = np.zeros((Nfits,3))
    Lvals = np.linspace(0,0.05,Nfits)
#    cfit = fit_model(p0,L)
    Ipmaps = np.zeros((Nfits,ARPES_dict['cube']['Y'][2],ARPES_dict['cube']['E'][2]))
    Ismaps = np.zeros((Nfits,ARPES_dict['cube']['Y'][2],ARPES_dict['cube']['E'][2]))
    for ii in range(Nfits):
        L = Lvals[ii]
        cfits[ii] = fit_model(p0,L)
        p0 = cfits[ii]
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
    
    
        args = [cfits[ii,1],cfits[ii,2],cfits[ii,0],L,Eg,txy,mG,b,cpar]
#    args = [Ef,Exy,p_nem,L,Eg,txy,mG,b,cpar]
        TB = build_TB(args)
#            _ = TB.solve_H()
#    TB.plotting(-0.25,0.1)
    
    
    
#    
        expmt = ARPES.experiment(TB,ARPES_dict)
        expmt.datacube(ARPES_dict)
        ARPES_dict['pol'] = np.array([0.707,0,0.707])
        Ip,Igp = expmt.spectral(ARPES_dict)
        ARPES_dict['pol'] = np.array([0,1,0])
        Is,Igs = expmt.spectral(ARPES_dict)
    
        y = np.linspace(*ARPES_dict['cube']['Y'])
        w = np.linspace(*ARPES_dict['cube']['E'])
        Y,W = np.meshgrid(y,w)
        fig = plt.figure()
        ax1= fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1.pcolormesh(Y,W,Igs[:,0,:].T,cmap=cm.Greys)
        ax2.pcolormesh(Y,W,Igp[:,0,:].T,cmap=cm.Greys)
        ax1.set_title('spol L:{:0.03f}eV,\nOO:{:0.03f}eV'.format(L,cfits[ii,0]))
        ax2.set_title('ppol L:{:0.03f}eV,\nOO:{:0.03f}eV'.format(L,cfits[ii,0]))
        ax1.set_xlabel('Momentum (1/A)')
        ax2.set_xlabel('Momentum (1/A)')
        ax1.set_ylabel('Energy (eV)')
        ax1.set_ylim(-0.2,0.05)
        ax2.set_ylim(-0.2,0.05)
        plt.tight_layout()
        plt.savefig('FeSe_SOC_maps/D2_FeSe_Imaps_37eV_120K_{:d}_L_{:0.04f}_OO_{:0.04f}.png'.format(ii,L,cfits[ii,0]),dpi=300,transparent=False)
        Ismaps[ii,:,:] = Igs[:,0,:]
        Ipmaps[ii,:,:] = Igp[:,0,:]
        
        index = np.where(abs(w+0.028)==abs(w+0.028).min())[0][0]
        mdcs = np.zeros((Nfits,len(y)))
        for ii in range(Nfits):
            mdcs[ii,:] = Ipmaps[ii,:,index]
            
        Ymesh,Lmesh = np.meshgrid(y,Lvals)
            
        fig2 = plt.figure()
        ax3 = fig2.add_subplot(121)
        ax4 = fig2.add_subplot(122)
        ax3.plot(cfits[:,0],Lvals)
        ax4.pcolormesh(Ymesh,Lmesh,mdcs,cmap=cm.Greys)
        ax3.set_xlabel('Orbital Order (eV)')
        ax3.set_ylabel('Spin-Orbit Coupling (eV)')
        ax4.set_xlabel('Momentum (1/A)')
        plt.tight_layout()
        plt.savefig('FeSe_SOC_maps/FeSe_ppol_mdcs_EB_28meV.png',dpi=300,transparent=True)
        
        
    with open('FeSe_SOC_maps/FeSe_params.txt','w') as tofile:
        tofile.write('LSOC,OO,EF,Exy\n')
        for ii in range(Nfits):
            tofile.write('{:0.06f},{:0.06f},{:0.06f},{:0.06f}\n'.format(Lvals[ii],cfits[ii,0],cfits[ii,1],cfits[ii,2]))
            
    tofile.close()
#    expmt.plot_gui(ARPES_dict)
##    
    
    
    
    
#