#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


Created on Tue Feb 27 08:01:41 2018

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

import numpy as np
import matplotlib.pyplot as plt
import build_lib
import ARPES_lib as ARPES
import scipy.ndimage as nd
import operator_library as ops
import Tk_plot



def gen_spr(n,a,c,plt_latt):
    v = np.array([[n*a,n*a,0],[n*a,-n*a,0],[0,0,c]])
    
    
    pts = []
    for i in range(4*n**2):
        x,y=int(i/(2*n)),int(i%(2*n))
        if x>=abs(y):
            tmp = np.array([x*a,y*a,0])
            if np.dot(tmp,v[0])<np.dot(v[0],v[0]) and np.dot(tmp,v[1])<np.dot(v[0],v[0]):
                pts.append(tmp)
            
                tmp2 = np.array([x*a,-y*a,0])
                if np.linalg.norm(tmp2-tmp)>0:
                    pts.append(tmp2)
    pts = np.array(pts)
    
    if plt_latt:
        corners =  np.array([v[0],v[1],np.zeros(3),v[0]+v[1]])
        plt.figure()
        plt.scatter(corners[:,0],corners[:,1],c='red')
        plt.scatter(pts[:,0],pts[:,1])

    return pts,v

def rand_AFM(D,pts):
    '''
    Randomly distribute the AFM-ordering over the supercell
    '''
    
    order = [-D]*len(pts)
    flipped = []
    for i in range(int(len(pts)/2)):
        unique = False
        while unique==False:
            pos = int(np.floor(len(pts)*np.random.random()))
            if pos not in flipped:
                order[pos] = D
                flipped.append(pos)
                unique = True
        unique = False
    order_dn = [-1*oi for oi in order]
    order = np.array(order+order_dn)
    
    return order
    
def stripe_AFM(D,pts):
    y_set = sorted(list(set([p[1] for p in pts])))
    od = {y_set[i]:D*(-1+2*(i%2)) for i in range(len(y_set))}
    order = [od[p[1]] for p in pts]
    order_dn = [-1*oi for oi in order]
    
    return np.array(order+order_dn)

def total_AFM(D,pts):
    
    y_set = sorted(list(set([p[1] for p in pts])))
    x_set = sorted(list(set([p[0] for p in pts])))
    
    dx = x_set[1]-x_set[0]
    dy = y_set[1]-y_set[0]
    
    order = [D*(-1+2*((p[0]/dx+p[1]/dy)%2)) for p in pts]
    
    order_dn = [-1*oi for oi in order]
    
    return np.array(order+order_dn)
    
    
            

if __name__=="__main__":
    
    
    a,c =  5.0,10.0
    n = 3
    p,avec = gen_spr(n,a,c,0)
    
    to = -1
    SK = {"020":0,"002200S":to}
    CUT,REN,OFF,TOL=a*1.1,1,0.0,0.001
    G2,G,M,X,Y,M2=np.array([1,1,0]),np.zeros(3),np.array([0.5,0.5,0.0]),np.array([0.5,0.0,0.0]),np.array([0.0,0.5,0.0]),np.array([0.5,-0.5,0])
	

    spin = {'soc':True,'lam':{0:0.0,1:0.0}}
    
    slab_dict = {'bool':False,
                'hkl':np.array([0,0,1]),
                'cells':20,
                'buff':8,
                'term':0,
                'avec':avec}

    Bd = {'atoms':[0]*len(p),
			'Z':{0:3},
			'orbs':[["20"]]*len(p),
			'pos':[p[ii] for ii in range(len(p))],
            'slab':slab_dict}

    Kd = {'type':'F',
			'pts':[M,G,M2],
			'grain':200,
			'labels':['X','$\Gamma$','M','X']}


    Hd = {'type':'SK',
			'V':SK,
          'avec':avec,
			'cutoff':CUT,
			'renorm':REN,
			'offset':OFF,
			'tol':TOL,
			'so':spin['soc']}
 
    	#####
    Bd = build_lib.gen_basis(Bd,spin)
    Kobj = build_lib.gen_K(Kd,avec)
    TB = build_lib.gen_TB(Bd,Hd,Kobj)
    #ADD AFM ORDER TO HAMILTONIAN
    D = 0.5
#    order = rand_AFM(D,p)
#    order = stripe_AFM(D,p)
    order = total_AFM(D,p)
    cols = ['r' if np.sign(o)>0 else 'b' for o in order]
    fig = plt.figure()
    plt.scatter(p[:,0],p[:,1],c=cols)
    
    for ii in range(len(TB.mat_els)):
        if TB.mat_els[ii].i==TB.mat_els[ii].j:
            TB.mat_els[ii].H.append([0,0,0,order[TB.mat_els[ii].i]])
#    
    TB.solve_H()
    TB.plotting(-6,6)
#    O = ops.LdotS(TB,axis=None,vlims=(-0.5,0.5),Elims=(-0.5,0.5))
##    
####    
#    ARPES_dict={'cube':{'X':[-0.62,0.62,201],'Y':[-0.62,0.62,201],'kz':0.0,'E':[-5.0,0.6,300]},
#                'SE':[0.02,0.00],
#                'directory':'C:\\Users\\rday\\Documents\\TB_ARPES\\2018\\TB_ARPES_2018\\FeSe',
#                'hv': 35,
#                'pol':np.array([0,np.sqrt(0.5),-np.sqrt(0.5)]),
#                'mfp':7.0,
#                'resolution':{'E':0.1,'k':0.003},
#                'T':[False,10.0],
#                'W':4.0,
#                'angle':0.0,
#                'spin':None,
#                'slice':[False,-0.2]}
#
##
##    
#
#    expmt = ARPES.experiment(TB,ARPES_dict)
#    expmt.datacube(ARPES_dict)
##    _,_ = expmt.spectral(ARPES_dict)
#    expmt.plot_gui(ARPES_dict)
####
####
