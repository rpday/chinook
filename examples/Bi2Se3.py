# -*- coding: utf-8 -*-
"""
Created on Fri Feb 02 20:05:23 2018

@author: rday
"""
import sys
sys.path.append('C:/Users/rday/Documents/TB_ARPES/2018/TB_ARPES_2018/TB_ARPES-master/')


import numpy as np
import matplotlib.pyplot as plt
import ubc_tbarpes.build_lib as build_lib
import ubc_tbarpes.ARPES_lib as ARPES
import scipy.ndimage as nd
import ubc_tbarpes.operator_library as ops
from scipy.optimize import curve_fit

def lin(x,a,b):
    return x*a+b

def surface_mod(Emax,length,TB):
    H_tmp = TB.mat_els.copy()
    for hi in H_tmp:
        if hi.i==hi.j:
            for hel in hi.H:
                if np.sqrt(hel[0]**2+hel[1]**2+hel[2]**2)==0.0:
                    #onsite
                    Hval = hel[-1]-Emax*np.exp(-abs(TB.basis[hi.i].depth)/length)
                    hel[-1]=Hval
    return H_tmp



if __name__=="__main__":
    a,c =  4.1141,28.64704 
    avec = np.array([[0,a/np.sqrt(3),c/3.],[-a/2.,-a/np.sqrt(12),c/3.],[a/2.,-a/np.sqrt(12),c/3.]])
    

    G,Z,F,L = np.array([0,0,0]),np.array([0.5,0.5,0.5]),np.array([0.5,0.5,0]),np.array([0,0,0.5])

    G,K,M = np.array([0.0,0.0,0.0],float),0.2*np.array([0.0,4*np.pi/3./a,0.0],float),0.2*np.array([2*np.pi/np.sqrt(3)/a,0.0,0.0])

    nu,mu=0.792,0.399
    
    SK1 = {"060":-10.7629,"061":0.2607,"140":-10.9210,"141":-1.5097,"240":-13.1410,"241":-1.1893,\
           "016400S":-0.6770,"016401S":2.0774,"016410S":-0.4792,"016411S":2.0595,"016411P":-0.4432,\
           "026400S":-0.2410,"026401S":-0.2012,"026410S":-0.0193,"026411S":2.0325,"026411P":-0.5320,\
           "114400S":-0.3326,"114401S":-0.0150,"114411S":0.9449,"114411P":-0.1050}

    SK2 = {"006600S":0.2212,"006601S":-0.3067,"006611S":0.3203,"006611P":-0.0510,\
           "114400S":-0.0640,"114401S":0.2833,"114411S":0.3047,"114411P":-0.0035,\
           "224400S":-0.0878,"224401S":-0.2660,"224411S":-0.1486,"224411P":-0.0590}

    SK3 = {"124400S":0.0229,"124401S":-0.0318,"124410S":-0.0778,"124411S":-0.0852,"124411P":0.0120,\
           "006600S":-0.0567,"006601S":-0.2147,"006611S":0.1227,"006611P":-0.0108,\
           "016400S":0.0333,"016401S":-0.0047,"016410S":0.2503,"016411S":-0.1101,"016411P":0.1015} #switched 016410S from 104601S--should be the exact same#also changed 214401S to 124401S
    
    SK_list = [SK1,SK2,SK3]
    CUT = [3.4,4.2,4.75]
    REN,OFF,TOL=1,0.4,0.001


    spin = {'bool':True,'soc':True,'lam':{0:2.066*2./3,1:0.3197*2./3,2:0.3632*2./3}}
    
    Bd = {'atoms':[1,0,2,0,1],
			'Z':{0:83,1:34,2:34},
			'orbs':[["40","41x","41y","41z"],["60","61x","61y","61z"],["40","41x","41y","41z"],["60","61x","61y","61z"],["40","41x","41y","41z"]],
			'pos':[-nu*(avec[0]+avec[1]+avec[2]),-mu*(avec[0]+avec[1]+avec[2]),np.array([0.0,0.0,0.0]),mu*(avec[0]+avec[1]+avec[2]),nu*(avec[0]+avec[1]+avec[2])], #orbital positions relative to origin
            'spin':spin}

    Kd = {'type':'A',
          'avec':avec,
			'pts':[K+np.array([0,0,10.0]),G+np.array([0,0,10.0]),M+np.array([0,0,10.0])],
			'grain':30,
			'labels':['K','$\Gamma$','M']}


    Hd = {'type':'SK',
          'V':SK_list,
          'avec':avec,
			'cutoff':CUT,
			'renorm':REN,
			'offset':OFF,
			'tol':TOL,
			'spin':spin}
    
    slab_dict = {'avec':avec,
      'miller':np.array([1,1,1]),
      'thick':10, 
      'vac':30,
      'fine':(-2,2),
      'termination':(1,1)}
 

    ARPES_dict={'cube':{'X':[-0.08,0.08,5],'Y':[-0.08,0.08,5],'kz':5.0,'E':[-0.55,0.05,100]},
                'SE':[0.005,0.0],
                'directory':'C:\\Users\\rday\\Documents\\TB_ARPES\\2018\\TB_ARPES_2018\\FeSe',
                'hv': 21.2,
                'pol':np.array([0,1,0]),
                'mfp':7.0,
                'slab':True,
                'resolution':{'E':0.01,'k':0.01},
                'T':[False,10.0],
                'W':0.0,
                'angle':0.0,
                'spin':None,
                'slice':[False,0.0]}



    	#####
    Bd = build_lib.gen_basis(Bd)

    Kobj = build_lib.gen_K(Kd)
    TB = build_lib.gen_TB(Bd,Hd,Kobj,slab_dict)
#    
#    theta = np.linspace(0,2*np.pi,100)
#    k = 0.02*np.array([[np.cos(th),np.sin(th),0] for th in theta])
#    
#    TB.Kobj.kpts = k
#    TB.Kobj.kcut = theta
#    TB.Kobj.kcut_brk = [0,2*np.pi]
    
    
    TB.solve_H()
    TB.plotting(-1.5,1.5)
    
         
    ARPES_expmt = ARPES.experiment(TB,ARPES_dict)
#    ARPES_expmt.plot_gui(ARPES_dict)
##
    
#    TB.mat_els = surface_mod(-0.22,20,TB)
#    
#    TB.solve_H()
#    TB.plotting(-1.5,1.5)
#    Sxmat = ops.S_vec(len(TB.basis),np.array([1,0,0]))
#    Symat = ops.S_vec(len(TB.basis),np.array([0,1,0]))
#    Sproj = ops.surface_proj(TB.basis,10)
##    print(Sxmat.max(),Symat.max(),Sproj.max())
#    Sysurf = np.dot(Symat,Sproj)
#    Sxsurf = np.dot(Sxmat,Sproj)
#    Sx = ops.O_path(Sxsurf,TB,TB.Kobj,vlims=(-0.1,0.1),Elims=(-1.5,1.5),degen=True)
#    Sy = ops.O_path(Sysurf,TB,TB.Kobj,vlims=(-0.25,0.25),Elims=(-1.5,1.5),degen=True)



#    TBn = TB.copy()
#    tmpH = TBn.mat_els.copy()
#    npts = 10
#    vfK=np.zeros(npts)
#    vfM = np.zeros(npts)
#    DP=np.zeros(npts)
#    klin = np.linalg.norm(TB.Kobj.kpts,axis=1)
#    zvals = np.zeros(npts)
#    Ig = np.zeros((npts,ARPES_dict['cube']['Y'][2]))
#    w = np.linspace(*ARPES_dict['cube']['E'])
#    for i in range(npts):
#        zf = 1.+0.005*(i-npts/2)
#        zvals[i]=zf
#        print(zf,TB.avec[:,2])
#        TBn.avec[:,2]=zf*TB.avec[:,2]
#        print(TBn.avec)
#
#    
#        for p in TBn.basis:
#            p.pos[2]*=zf
#    
#    
#        for ti in tmpH:
#            for hi in ti.H:
#                nz = hi[2]*zf
#                nR = np.linalg.norm(np.array([hi[0],hi[1],nz]))
#                oR = np.linalg.norm(np.array(hi[0:3]))
#                nH = hi[3]*np.exp(-(nR-oR)/5)
#                hi[2]=nz
#                hi[3]=nH
#    
#    
#        TBn.mat_els = tmpH
#        
#        _=TBn.solve_H()
#        DP[i]=TBn.Eband[30,222]
#        wi = np.where(abs(w-DP[i]-0.2)==abs(w-DP[i]-0.2).min())[0][0]
#        dispM = TBn.Eband[30:40,222]
#        dispK = TBn.Eband[20:30,222]
#        kvals = klin[30:40]
#        p0 = (5,DP[i])
#        c,_ = curve_fit(lin,kvals,dispM,p0=p0)
#        vfM[i]=c[0]
#        p0 = (-5,DP[i])
#        kvals = klin[20:30]
#        c,_=curve_fit(lin,kvals,dispK,p0=p0)
#        vfK[i]=c[0]
#        TBn.plotting(-0.75,0.75)
#        plt.savefig('Bi2Se3_z_{:0.04f}.jpg'.format(TBn.avec[2,2]))
#        ARPES_expmt = ARPES.experiment(TBn,ARPES_dict)
#        ARPES_expmt.datacube(ARPES_dict)
#        _,Ig_tmp = ARPES_expmt.spectral(ARPES_dict)
##        plt.figure()
##        plt.pcolormesh(Ig_tmp[:,0,:])
##        plt.plot(k,Ig_tmp[:,0,wi])
#        Ig[i] = Ig_tmp[:,0,wi]
#
#        TBn = TB.copy()
#        tmpH = TBn.mat_els.copy()
#
#    
#    k = np.linspace(*ARPES_dict['cube']['Y'])
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    for i in range(npts):
#        
#        ax.plot(k,Ig[i])
#        
#        
#    def lorentz(x,A,wi,wo):
#        return A/((x-wo)**2+(wi/2)**2)
#    
#    
#    def lorentz2(x,A1,A2,wo1,wo2,wi1,wi2,B):
#        return lorentz(A1,wi1,wo1) + lorentz(A2,wi2,wo2)+B
#    
#    p0 = (0.00001,0.00001,-0.06,0.06,0.002,0.002,0.0)
#    fits = np.zeros((npts,len(p0)))
#    for i in range(npts):
#        fits[i,:],_ = curve_fit(lorentz2,k,Ig[i],p0=p0)
#        p0 = fits[i,:]
#        
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    ax.plot(zvals,fits[:,0]/fits[:,1])
#    ax.legend(['Amplitude Left Peak','Amplitude Right Peak'])
#    ax.set_xlabel('Slab Thickness (A)')
#    ax.set_ylabel('Peak Amplitude (a.u.)')
#    ax.set_title('Bi2Se3 ARPES Intensity Variation under C-lattice Oscillation')
    #    fig = plt.figure()
#    plt.plot(zvals,DP)
#    plt.figure()
#    plt.plot(zvals,abs(vfM))
#    plt.plot(zvals,abs(vfK))



#    
 ###    
##    sigma = {(0,1):2.7,(1,1):8.0,(2,1):8.0,(0,0):1.0,(1,0):1.0,(2,0):1.0}
##    for bi in TB.basis:
##        bi.sigma = sigma[(bi.atom,bi.l)]
#    
##    





    
   
###
##
##    
##
##    
##    
#    
#    

#	#####