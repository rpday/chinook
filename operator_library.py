# -*- coding: utf-8 -*-
"""
Created on Thu Feb 01 08:56:54 2018

@author: rday
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import H_library as Hlib
import TB_lib as TBlib
from matplotlib import rc 

'''
Library for different operators of possible interest in calculating, diagnostics, etc for a material of interest

'''

def LSmat(TB,axis=None):
    '''
    Generate an arbitary L.S type matrix for a given basis. Uses same Yproj as 
    the HSO in the H_library, but is otherwise different. 
    This is generic to all l, except the normal_order, specifically defined for l=1,2...should generalize
    Otherwise, this structure holds for all l!
    The user gives an 'axis'--if None, then just compute the L.S matrix. Otherwise, the LiSi matrix is computed
    with i the axis index. To do this, a linear combination of L+S+,L-S-,L+S-,L-S+,LzSz terms are used to compute
    In the factos dictionary, the weight of these terms is defined. The keys are tuples of (L+/-/z,S+/-/z) in a bit
    of a cryptic way. For L, range (0,1,2) ->(-1,0,1) and for S range (-1,0,1) = S1-S2 with S1/2 = +/- 1 here
    
    L+,L-,Lz matrices are defined for each l shell in the basis, transformed into the basis of cubic harmonics.
    The nonzero terms will then just be used along with the spin and weighted by the factor value, and slotted into 
    a len(basis)xlen(basis) matrix HSO
    args:
        TB -- tight-binding object, as defined in TB_lib.py
        axis --axis for calculation as either 'x','y','z',None, or float (angle in the x-y plane)
    return:
        HSO (len(basis)xlen(basis)) numpy array of complex float
    '''
    Md = Hlib.Yproj(TB.basis)
    normal_order = {1:{'x':0,'y':1,'z':2},2:{'xz':0,'yz':1,'xy':2,'ZR':3,'XY':4}}
    factors = {(2,-1):0.5,(0,1):0.5,(1,0):1.0}
    L,al = {},[]
    HSO = np.zeros((len(TB.basis),len(TB.basis)),dtype=complex)
    for o in TB.basis:
        if (o.atom,o.l) not in al:
            al.append((o.atom,o.l))
            M = Md[(o.atom,o.l)]
            Mp = np.linalg.inv(M)
            L[(o.atom,o.l)] = [np.dot(Mp,np.dot(Lm(o.l),M)),np.dot(Mp,np.dot(Lz(o.l),M)),np.dot(Mp,np.dot(Lp(o.l),M))]
    if axis is not None:
        try:
            ax = float(axis)
            factors = {(0,1):0.5,(2,-1):0.5,(2,1):0.5*np.exp(-1.0j*2*ax),(0,-1):0.5*np.exp(1.0j*2*ax)}
        except ValueError:
            if axis=='x':
                factors = {(0,1):0.25,(2,-1):0.25,(2,1):0.25,(0,-1):0.25}
            elif axis=='y':
                factors = {(0,1):-0.25,(2,-1):-0.25,(2,1):0.25,(0,-1):0.25}
            elif axis=='z':
                factors = {(1,0):1.0}
            else:
                print('Invalid axis entry')
                return None
    for o1 in TB.basis:
        for o2 in TB.basis:
            if o1.index<=o2.index:
                LS_val = 0.0
                if np.linalg.norm(o1.pos-o2.pos)<0.0001 and o1.l==o2.l and o1.n==o2.n:
                    inds = (normal_order[o1.l][o1.label[2:]],normal_order[o2.l][o2.label[2:]])
                    
                    ds = (o1.spin-o2.spin)/2.
                    if ds==0:
                        s=0.5*np.sign(o1.spin)
                    else:
                        s=1.0
                    for f in factors:
                        if f[1]==ds:
                            LS_val+=factors[f]*L[(o1.atom,o1.l)][f[0]][inds]*s
                    HSO[o1.index,o2.index]+=LS_val
                    if o1.index!=o2.index:
                        HSO[o2.index,o1.index]+=np.conj(LS_val)
    return HSO
        

def Lp(l):
    '''
    L+ operator in the l,m_l basis, organized with (0,0) = |l,l>, (2*l,2*l) = |l,-l>
    The nonzero elements are on the upper diagonal
    arg: l int orbital angular momentum
    return M: numpy array (2*l+1,2*l+1) of real float
    '''
    M = np.zeros((2*l+1,2*l+1))
    r = np.arange(0,2*l,1)
    M[r,r+1]=1.0
    vals = [0]+[np.sqrt(l*(l+1)-(l-m)*(l-m+1)) for m in range(1,2*l+1)]
    M = M*vals
    return M

def Lm(l):
    '''
    L- operator in the l,m_l basis, organized with (0,0) = |l,l>, (2*l,2*l) = |l,-l>
    The nonzero elements are on the upper diagonal
    arg: l int orbital angular momentum
    return M: numpy array (2*l+1,2*l+1) of real float
    '''
    M = np.zeros((2*l+1,2*l+1))
    r = np.arange(1,2*l+1,1)
    M[r,r-1]=1.0
    vals = [np.sqrt(l*(l+1)-(l-m)*(l-m-1)) for m in range(0,2*l)]+[0]
    M = M*vals
    return M

def Lz(l):
    '''
    Lz operator in the l,m_l basis
    arg: l int orbital angular momentum
    retun numpy array (2*l+1,2*l+1)
    '''
    return np.identity(2*l+1)*np.array([l-m for m in range(2*l+1)])



def O_path(O,TB,Kobj=None,axis=None,vlims=(0,0),Elims=(0,0)):
    if np.shape(O)!=(len(TB.basis),len(TB.basis)):
        print('ERROR! Ensure your operator has the same dimension as the basis.')
        return None
    try:
        np.shape(TB.Evec)
    except AttributeError:
        try:
            len(TB.Kobj.kpts)
            TB.solve_H()
        except AttributeError:    
            TB.Kobj = Kobj
            try:
                TB.solve_H()
            except TypeError:
                print('ERROR! Please include a K-object, or diagonalize your tight-binding model over a k-path first to initialize the eigenvectors')
                return None
    O_vals = np.zeros(np.shape(TB.Eband))
    fig = plt.figure()
    ax=fig.add_subplot(111)
    plt.axhline(y=0,color='grey',lw=1,ls='--')
    rc('font',**{'family':'serif','serif':['Palatino'],'size':20})
    rc('text',usetex = False)
    for b in TB.Kobj.kcut_brk:
        plt.axvline(x = b,color = 'grey',ls='--',lw=1.0)
        
    for e in range(len(TB.Evec)):
        for p in range(len(TB.basis)):
            O_vals[e,p] = np.real(np.dot(np.conj(TB.Evec[e,:,p]),np.dot(O,TB.Evec[e,:,p])))#operator expectation values MUST be real--they correspond to physical observables
            
    if vlims==(0,0):
        vlims = (O_vals.min()-(O_vals.max()-O_vals.min())/10.0,O_vals.max()+(O_vals.max()-O_vals.min())/10.0)
    if Elims==(0,0):
        Elims = (TB.Eband.min()-(TB.Eband.max()-TB.Eband.min())/10.0,TB.Eband.max()+(TB.Eband.max()-TB.Eband.min())/10.0)
        
    for p in range(len(TB.basis)):
        plt.plot(TB.Kobj.kcut,TB.Eband[:,p],color='navy',lw=1.0)
        O_line=plt.scatter(TB.Kobj.kcut,TB.Eband[:,p],c=O_vals[:,p],cmap=cm.Spectral,marker='.',lw=0,s=60,vmin=vlims[0],vmax=vlims[1])
    plt.axis([TB.Kobj.kcut[0],TB.Kobj.kcut[-1],Elims[0],Elims[1]])
    plt.xticks(TB.Kobj.kcut_brk,TB.Kobj.labels)
    plt.colorbar(O_line,ax=ax)
    plt.ylabel("Energy (eV)")
    
    
    return O_vals

def LdotS(TB,axis=None,Kobj=None,vlims=(0,0),Elims=(0,0)):
    HSO = LSmat(TB,axis)
    O = O_path(HSO,TB,axis,Kobj,vlims,Elims)
    return O
    
    




def is_numeric(a):
    '''
    Quick check if object is numeric
    '''
    if a is not None:
        
        try:
            float(a)
            return True
        except ValueError:
            return False
    else:
        return False
    
    
