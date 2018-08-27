# -*- coding: utf-8 -*-
"""
Created on Thu Feb 01 08:56:54 2018

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
import matplotlib.cm as cm
import ubc_tbarpes.H_library as Hlib
import ubc_tbarpes.TB_lib as TBlib
#from matplotlib import rc 
import ubc_tbarpes.klib as K_lib

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
    normal_order = {0:{'':0},1:{'x':0,'y':1,'z':2},2:{'xz':0,'yz':1,'xy':2,'ZR':3,'XY':4}}
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



def Lz_mat(TB):
    '''
    Generate an arbitary Lz type matrix for a given basis. Uses same Yproj as 
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
    return:
        HL (len(basis)xlen(basis)) numpy array of complex float
    '''
    Md = Hlib.Yproj(TB.basis)
    normal_order = {0:{'':0},1:{'x':0,'y':1,'z':2},2:{'xz':0,'yz':1,'xy':2,'ZR':3,'XY':4}}
    L,al = {},[]
    HL = np.zeros((len(TB.basis),len(TB.basis)),dtype=complex)
    for o in TB.basis:
        if (o.atom,o.l) not in al:
            al.append((o.atom,o.l))
            M = Md[(o.atom,o.l)]
            Mp = np.linalg.inv(M)
            L[(o.atom,o.l)] = np.dot(Mp,np.dot(Lz(o.l),M))

    for o1 in TB.basis:
        for o2 in TB.basis:
            if o1.index<=o2.index:
                L_val = 0.0
                if np.linalg.norm(o1.pos-o2.pos)<0.0001 and o1.l==o2.l and o1.n==o2.n:
                    inds = (normal_order[o1.l][o1.label[2:]],normal_order[o2.l][o2.label[2:]])
                    
                    ds = (o1.spin-o2.spin)/2.
                    if ds==0:
                        L_val+=L[(o1.atom,o1.l)][inds]
                    HL[o1.index,o2.index]+=L_val
                    if o1.index!=o2.index:
                        HL[o2.index,o1.index]+=np.conj(L_val)
    return HL
        

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


def fatbs(proj,TB,Kobj=None,vlims=(0,0),Elims=(0,0),degen=False):
    '''
    Fat band projections. User denotes which orbital index projection is of interest
    
    '''
    proj = np.array(proj)
    
    O = np.identity(len(TB.basis))
    
    if len(np.shape(proj))<2:
        tmp = np.zeros((len(proj),2))
        tmp[:,0] = proj
        tmp[:,1] = 1.0
        proj = tmp
    pvec = np.zeros(len(TB.basis),dtype=complex)
    try:
        pvec[np.real(proj[:,0]).astype(int)] = proj[:,1]
        O = O*pvec
    
        Ovals = O_path(O,TB,Kobj,vlims,Elims,degen=degen)
    except ValueError:
        print('projections need to be passed as list or array of type [index,projection]')
    
    
    
    return Ovals
    
    


def O_path(O,TB,Kobj=None,vlims=(0,0),Elims=(0,0),degen=False):
    '''Compute and plot the expectation value of an user-defined operator along a k-path
        Option of summing over degenerate bands (for e.g. fat bands) with degen boolean flag
        
        args: O -- matrix representation of the operator (numpy array len(basis), len(basis) of complex float)
            TB -- Tight binding object from TB_lib
            Kobj -- Kobject -- if not defined, use the TB object's Kobject
            vlims -- limits for the colourscale (optional argument)
            Elims -- limit for energy limits in plot (optional)
            degen -- degenerate sum flag

        return -- the numpy array of expectation values
    
    '''
    
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
            
    O_vals = np.zeros((int(np.shape(TB.Eband)[0]),int(np.shape(TB.Eband)[1]/int(2 if degen else 1))))

    fig = plt.figure()
    ax=fig.add_subplot(111)
#    plt.axhline(y=0,color='grey',lw=1,ls='--')
#    rc('font',**{'size':20})
#    rc('text',usetex = True)
    for b in TB.Kobj.kcut_brk:
        plt.axvline(x = b,color = 'grey',ls='--',lw=1.0)
        
    for e in range(len(TB.Evec)):
        for p in range(np.shape(O_vals)[1]):
            if degen:
                O_vals[e,p] = np.real(np.dot(np.conj(TB.Evec[e,:,2*p]),np.dot(O,TB.Evec[e,:,2*p])))#operator expectation values MUST be real--they correspond to physical observables
                O_vals[e,p] += np.real(np.dot(np.conj(TB.Evec[e,:,2*p+1]),np.dot(O,TB.Evec[e,:,2*p+1])))
            else:
                O_vals[e,p] = np.real(np.dot(np.conj(TB.Evec[e,:,p]),np.dot(O,TB.Evec[e,:,p])))
                
    if vlims==(0,0):
        vlims = (O_vals.min()-(O_vals.max()-O_vals.min())/10.0,O_vals.max()+(O_vals.max()-O_vals.min())/10.0)
    if Elims==(0,0):
        Elims = (TB.Eband.min()-(TB.Eband.max()-TB.Eband.min())/10.0,TB.Eband.max()+(TB.Eband.max()-TB.Eband.min())/10.0)
        
    for p in range(np.shape(O_vals)[1]):
        plt.plot(TB.Kobj.kcut,TB.Eband[:,(2 if degen else 1)*p],color='navy',lw=0.1)
        O_line=plt.scatter(TB.Kobj.kcut,TB.Eband[:,(2 if degen else 1)*p],c=O_vals[:,p],cmap=cm.RdBu,marker='.',lw=0,s=50,vmin=vlims[0],vmax=vlims[1])
    plt.axis([TB.Kobj.kcut[0],TB.Kobj.kcut[-1],Elims[0],Elims[1]])
    plt.xticks(TB.Kobj.kcut_brk,TB.Kobj.labels)
    plt.colorbar(O_line,ax=ax)
    plt.ylabel("Energy (eV)")
    
    
    return O_vals


def O_surf(O,TB,ktuple,Ef,tol,vlims=(-1,1)):
    '''Compute and plot the expectation value of an user-defined operator over a surface of constant-binding energy
        Option of summing over degenerate bands (for e.g. fat bands) with degen boolean flag
        
        args: O -- matrix representation of the operator (numpy array len(basis), len(basis) of complex float)
            TB -- Tight binding object from TB_lib
            ktuple -- momentum range for mesh: ktuple[0] = (x0,xn,n),ktuple[1]=(y0,yn,n),ktuple[2]=kz
            vlims -- limits for the colourscale (optional argument)


        return -- the numpy array of expectation values
    
    '''
    coords,Eb,Ev=FS(TB,ktuple,Ef,tol)
    masked_Ev = np.array([Ev[int(coords[ei,2]/len(TB.basis)),:,int(coords[ei,2]%len(TB.basis))] for ei in range(len(coords))])
    Ovals = np.sum(np.conj(masked_Ev)*np.dot(O,masked_Ev.T).T,1)
    
    pts = np.array([[coords[ci,0],coords[ci,1],np.real(Ovals[ci])] for ci in range(len(coords))])
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.scatter(pts[:,0],pts[:,1],c=pts[:,2],cmap=cm.RdBu,s=200,vmin=vlims[0],vmax=vlims[1])
    plt.scatter(pts[:,0],pts[:,1],c='k',s=5)
    
    return pts


def surface_projection(TB,cutoff):
    
    M = np.identity(len(TB.basis))
    projs = np.array([np.exp(bi.depth/cutoff) for bi in TB.basis])
    M = M*projs
    return M
    
   
    
    
    


def FS(TB,ktuple,Ef,tol):
    x,y,z=np.linspace(*ktuple[0]),np.linspace(*ktuple[1]),ktuple[2]
    X,Y=np.meshgrid(x,y)
        
    k_arr,_ = K_lib.kmesh(0.0,X,Y,z)  

    blen = len(TB.basis)    

    TB.Kobj = K_lib.kpath(k_arr)
    _,_ = TB.solve_H()
    TB.Eband = np.reshape(TB.Eband,(np.shape(TB.Eband)[-1]*np.shape(X)[0]*np.shape(X)[1])) 
    pts = []
    for ei in range(len(TB.Eband)):
        if abs(TB.Eband[ei]-Ef)<tol: 
            inds = (int(np.floor(np.floor(ei/blen)/np.shape(X)[1])),int(np.floor(ei/blen)%np.shape(X)[1]))
            pts.append([X[inds],Y[inds],ei])
    
    pts = np.array(pts)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(pts[:,0],pts[:,1])
    return pts,TB.Eband,TB.Evec
    
    
####################SOME STANDARD OPERATORS FOLLOW HERE: ######################
    
    

def LdotS(TB,axis=None,Kobj=None,vlims=(0,0),Elims=(0,0)):
    HSO = LSmat(TB,axis)
    O = O_path(HSO,TB,Kobj,vlims,Elims)
    return O

def Sz(TB,Kobj=None,vlims=(0,0),Elims=(0,0)):
    Omat = Sz_mat(TB)
    O = O_path(Omat,TB,Kobj,vlims,Elims)
    return O

def Lz_path(TB,Kobj=None,vlims=(0,0),Elims=(0,0)):
    Omat = Lz_mat(TB)
    O = O_path(Omat,TB,Kobj,vlims,Elims)
    return O

def LzSz_path(TB,Kobj=None,vlims=(0,0),Elims=(0,0)):
    Omat = np.dot(Lz_mat(TB),Sz_mat(TB))
    O = O_path(Omat,TB,Kobj,vlims,Elims)
    return O

def Jz_path(TB,Kobj=None,vlims=(0,0),Elims=(0,0)):
    Omat = Lz_mat(TB)+Sz_mat(TB)
    O = O_path(Omat,TB,Kobj,vlims,Elims)
    return O

def Sz_mat(TB):
    M = np.identity(len(TB.basis))
    els = np.array([-0.5 if j<len(TB.basis)/2 else 0.5 for j in range(len(TB.basis))])
    M*=els
    return M

    




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
    
    
