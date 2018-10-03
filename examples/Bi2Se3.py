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
			'pts':[K,G,M],
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
 

    ARPES_dict={'cube':{'X':[-0.08,0.08,60],'Y':[-0.08,0.08,60],'kz':0.0,'E':[-0.55,0.05,100]},
                'SE':[0.005,0.0],
                'directory':'C:\\Users\\rday\\Documents\\TB_ARPES\\2018\\TB_ARPES_2018\\FeSe',
                'hv': 21.2,
                'pol':np.array([0,1,0]),
                'mfp':10.0,
                'slab':True,
                'resolution':{'E':0.01,'k':0.01},
                'T':[True,10.0],
                'W':4.0,
                'angle':0.0,
                'spin':None,
                'slice':[False,0.0]}



    	#####
    Bd = build_lib.gen_basis(Bd)

    Kobj = build_lib.gen_K(Kd)
    TB = build_lib.gen_TB(Bd,Hd,Kobj,slab_dict)
    
    
     TB.solve_H()
    TB.plotting(-1.5,1.5)
#    
 ###    
##    sigma = {(0,1):2.7,(1,1):8.0,(2,1):8.0,(0,0):1.0,(1,0):1.0,(2,0):1.0}
##    for bi in TB.basis:
##        bi.sigma = sigma[(bi.atom,bi.l)]
#    
##    


        
    ARPES_expmt = ARPES.experiment(TB,ARPES_dict)
    ARPES_expmt.plot_gui(ARPES_dict)
###
###
##
##    
##
##    
##    
#    
#    
##TB_og = TB.copy()
##TBn = TB_og.copy()
##tmpH = TBn.mat_els.copy()
##vfK=np.zeros(10)
##vfM = np.zeros(10)
##DP=np.zeros(10)
##zvals = np.zeros(10)
##for i in range(10):
##    print(TBn.avec)
##    zf = 1.+0.005*i
##    zvals[i]=zf
##    TBn.avec[:,2]=zf*TB_og.avec[:,2]
##
##    for p in TBn.basis:
##        p.pos[2]*=zf
##
##
##    for ti in tmpH:
##        for hi in ti.H:
##            nz = hi[2]*zf
##            nR = np.linalg.norm(np.array([hi[0],hi[1],nz]))
##            oR = np.linalg.norm(np.array(hi[0:3]))
##            nH = hi[3]*np.exp(-abs(nR-oR)/5)
##            hi[2]=nz
##            hi[3]=nH
##
##
##    TBn.mat_els = tmpH
##    
##    _=TBn.solve_H()
##    DP[i]=TBn.Eband[30,222]
##    dispM = TBn.Eband[30:40,222]
##    dispK = TBn.Eband[20:30,222]
##    kvals = klin[30:40]
#    p0 = (5,DP[i])
#    c,_ = curve_fit(lin,kvals,dispM,p0=p0)
#    vfM[i]=c[0]
#    p0 = (-5,DP[i])
#    c,_=curve_fit(lin,kvals,dispK,p0=p0)
#    vfK[i]=c[0]
#    TBn.plotting(-0.75,0.75)
#    plt.savefig('Bi2Se3_z_{:0.04f}.jpg'.format(TBn.avec[2,2]))
#    TBn = TB_og.copy()
#    tmpH = TBn.mat_els.copy()
#fig = plt.figure()
#plt.plot(zvals,DP)
#plt.figure()
#plt.plot(zvals,vfM)
#plt.plot(zvals,abs(vfK))
#	#####