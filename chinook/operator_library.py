# -*- coding: utf-8 -*-

#Created on Thu Feb 01 08:56:54 2018

#@author: ryanday
#MIT License

#Copyright (c) 2018 Ryan Patrick Day

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A ▲ICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.cm as cm
import chinook.klib as K_lib
import chinook.Ylm as Ylm

'''
Library for different operators of possible interest in calculating, diagnostics, etc for a material of interest

'''

def LSmat(TB,axis=None):
    '''
    Generate an arbitary L.S type matrix for a given basis. Uses same *Yproj* as 
    the *HSO* in the *chinook.H_library*, but is otherwise different, as it supports
    projection onto specific axes, in addition to the full vector dot product operator. 

    Otherwise, the LiSi matrix is computed with i the axis index. 
    To do this, a linear combination of L+S+,L-S-,L+S-,L-S+,LzSz terms are used to compute.
    
    In the factors dictionary, the weight of these terms is defined. 
    The keys are tuples of (L+/-/z,S+/-/z) in a bit
    of a cryptic way. For L, range (0,1,2) ->(-1,0,1) 
    and for S range (-1,0,1) = S1-S2 with S1/2 = +/- 1 here
    
    L+,L-,Lz matrices are defined for each l shell in the basis, 
    transformed into the basis of cubic harmonics.
    The nonzero terms will then just be used along with the spin and 
    weighted by the factor value, and slotted into 
    a len(basis)xlen(basis) matrix HSO
    
    *args*:

        - **TB**: tight-binding object, as defined in TB_lib.py
        
        - **axis**: axis for calculation as either 'x','y','z',None,
        or float (angle in the x-y plane)
        
    *return*:

        - **HSO**: (len(basis)xlen(basis)) numpy array of complex float
        
    ***
    '''
    
    Md = Ylm.Yproj(TB.basis)
    normal_order = {0:{'':0},1:{'x':0,'y':1,'z':2},2:{'xz':0,'yz':1,'xy':2,'ZR':3,'XY':4},3:{'z3':0,'xz2':1,'yz2':2,'xzy':3,'zXY':4,'xXY':5,'yXY':6}}
    factors = {(2,-1):0.5,(0,1):0.5,(1,0):1.0}
    L,al = {},[]
    HSO = np.zeros((len(TB.basis),len(TB.basis)),dtype=complex)
    for o in TB.basis:
        if (o.atom,o.n,o.l) not in al:
            al.append((o.atom,o.n,o.l))
            Mdn = Md[(o.atom,o.n,o.l,-1)]
            Mup = Md[(o.atom,o.n,o.l,1)]
            Mdnp = np.linalg.inv(Mdn)
            Mupp = np.linalg.inv(Mup)
            L[(o.atom,o.n,o.l)] = [np.dot(Mupp,np.dot(Lm(o.l),Mdn)),np.dot(Mdnp,np.dot(Lz(o.l),Mdn)),np.dot(Mdnp,np.dot(Lp(o.l),Mup))]
    if axis is not None:
        try:
            ax = float(axis)
            factors = {(0,1):0.25,(2,-1):0.25,(2,1):0.25*np.exp(-1.0j*2*ax),(0,-1):0.25*np.exp(1.0j*2*ax)}
        except ValueError:
            if axis=='x':
                factors = {(0,1):0.25,(2,-1):0.25,(2,1):0.25,(0,-1):0.25}
            elif axis=='y':
                factors = {(0,1):0.25,(2,-1):0.25,(2,1):-0.25,(0,-1):-0.25}
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
                            LS_val+=factors[f]*L[(o1.atom,o1.n,o1.l)][f[0]][inds]*s
                    HSO[o1.index,o2.index]+=LS_val
                    if o1.index!=o2.index:
                        HSO[o2.index,o1.index]+=np.conj(LS_val)
    return HSO

        

def Lp(l):
    
    '''
    
    L+ operator in the l,m_l basis, organized with 
    (0,0) = |l,l>, (2*l,2*l) = |l,-l>
    The nonzero elements are on the upper diagonal
    
    *arg*: 

        - **l**: int orbital angular momentum
        
    *return*:

        - **M**: numpy array (2*l+1,2*l+1) of real float
    ***
    '''
    
    M = np.zeros((2*l+1,2*l+1))
    r = np.arange(0,2*l,1)
    M[r,r+1]=1.0
    vals = [0]+[np.sqrt(l*(l+1)-(l-m)*(l-m+1)) for m in range(1,2*l+1)]
    M = M*vals
    return M

def Lm(l):
    
    '''
    
    L- operator in the l,m_l basis, organized with 
    (0,0) = |l,l>, (2*l,2*l) = |l,-l>
    The nonzero elements are on the upper diagonal
    
    *arg*:

        - **l**: int orbital angular momentum
    
    *return*:

        - **M**: numpy array (2*l+1,2*l+1) of real float
        
    ***
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
    
    *arg*:

        - **l**: int orbital angular momentum
        
    *return*:

        - numpy array (2*l+1,2*l+1)
        
    ***
    '''
    return np.identity(2*l+1)*np.array([l-m for m in range(2*l+1)])


def fatbs(proj,TB,Kobj=None,vlims=(0,1),Elims=(-1,1),degen=False):
    
    '''
    
    Fat band projections. User denotes which orbital index projection is of interest
    Projection passed either as an Nx1 or Nx2 array of float. If Nx2, first column is
    the indices of the desired orbitals, the second column is the weight. If Nx1, then
    the weights are all taken to be eqaul
    
    *args*:

        - **proj**: iterable of projections, to be passed as either a 1-dimensional
        with indices of projection only, OR, 2-dimensional, with the second column giving
        the amplitude of projection (for linear-combination projection)
            
        - **TB**: tight-binding object
            
    *kwargs*:

        - **Kobj**: Momentum object, as defined in *chinook.klib.py*
            
        - **vlims**: tuple of 2 float, limits of the colorscale for plotting, default to (0,1)
            
        - **Elims**: tuple of 2 float, limits of vertical scale for plotting
            
        - **degen**: bool, True if bands are degenerate, sum over adjacent bands
            
    *return*:

        - **Ovals**: numpy array of float, len(Kobj.kpts)*len(TB.basis)
        
    ***
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
    
        Ovals = None
    
    return Ovals
    
    


def O_path(Operator,TB,Kobj=None,vlims=(0,0),Elims=(-10,10),degen=False,plot=True,ax=None):
    
    '''
    
    Compute and plot the expectation value of an user-defined operator along a k-path
    Option of summing over degenerate bands (for e.g. fat bands) with degen boolean flag
        
    *args*:

        - **Operator**: matrix representation of the operator (numpy array len(basis), len(basis) of complex float)
          
        - **TB**: Tight binding object from TB_lib
            
    *kwargs*:
        
        - **Kobj**: Momentum object, as defined in *chinook.klib.py*
            
        - **vlims**: tuple of 2 float, limits of the colourscale for plotting, default to (0,0)
        if default value passed, will compute a reasonable range
            
        - **Elims**: tuple of 2 float, limits of vertical scale for plotting
            
        - **degen**: bool, True if bands are degenerate, sum over adjacent bands
        
        - **ax**: matplotlib Axes, option for plotting onto existing Axes

    *return*:

        - **O_vals**: the numpy array of float, (len Kobj x len basis) expectation values 
        
        - **ax**: matplotlib Axes, allowing for user to further modify
    
    ***
    '''
    
    if np.shape(Operator)!=(len(TB.basis),len(TB.basis)):
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
            
    #check if Hermitian
    
       
    right_product = np.einsum('ij,ljm->lim',Operator,TB.Evec)
    O_vals = np.einsum('ijk,ijk->ik',np.conj(TB.Evec),right_product)
    O_vals = np.real(O_vals) #any Hermitian operator must have real-valued expectation value--discard any imaginary component
    if degen:
        O_vals = degen_Ovals(O_vals,TB.Eband)


    rcParams.update({'font.size':14})
    
    if ax is None:
        fig = plt.figure()
        fig.set_tight_layout(False)
        ax=fig.add_subplot(111)

    for b in TB.Kobj.kcut_brk:
        plt.axvline(x = b,color = 'grey',ls='--',lw=1.0)
                    
    if np.sign(O_vals.min())<0 and np.sign(O_vals.max())>0:
        my_cmap = cm.RdBu
    else:
        my_cmap = cm.Blues
    
    if vlims==(0,0):
        vlims = (O_vals.min()-(O_vals.max()-O_vals.min())/10.0,O_vals.max()+(O_vals.max()-O_vals.min())/10.0)
    if Elims==(-10,10):
        Elims = (TB.Eband.min()-(TB.Eband.max()-TB.Eband.min())/10.0,TB.Eband.max()+(TB.Eband.max()-TB.Eband.min())/10.0)
    if plot:
        for p in range(np.shape(O_vals)[1]):

            plt.plot(TB.Kobj.kcut,TB.Eband[:,p],color='k',lw=0.2)
            O_line=plt.scatter(TB.Kobj.kcut,TB.Eband[:,p],c=O_vals[:,p],cmap=my_cmap,marker='.',lw=0,s=80,vmin=vlims[0],vmax=vlims[1])

        plt.axis([TB.Kobj.kcut[0],TB.Kobj.kcut[-1],Elims[0],Elims[1]])
        plt.xticks(TB.Kobj.kcut_brk,TB.Kobj.labels)
        plt.colorbar(O_line,ax=ax)
        plt.ylabel("Energy (eV)")
        
    return O_vals,ax


def degen_Ovals(Oper_exp,Energy):
    
    '''
    In the presence of degeneracy, we want to average over the
    evaluated orbital expectation values--numerically, the degenerate 
    subspace can be arbitrarily diagonalized during numpy.linalg.eigh. 
    All degeneracies are found, and the expectation values averaged.
    
    *args*:

        - **Oper_exp**: numpy array of float, operator expectations
        
        - **Energy**: numpy array of float, energy eigenvalues.
    
    ***
    '''
    
    O_copy = Oper_exp.copy()
    tol = 1e-9
    for ki in range(np.shape(Oper_exp)[0]):
        val = Energy[ki,0]
        start = 0
        counter = 1
        for bi in range(1,np.shape(Oper_exp)[1]):
            if abs(Energy[ki,bi]-val)<tol:
                counter+=1

            if abs(Energy[ki,bi]-val)>=tol or bi==(np.shape(Oper_exp)[1]-1):

                O_copy[ki,start:start+counter] = np.mean(O_copy[ki,start:start+counter])
                start = bi
                counter = 1
                val = Energy[ki,bi]
    return O_copy                

def O_surf(O,TB,ktuple,Ef,tol,vlims=(0,0),ax=None):
    
    '''
    
    Compute and plot the expectation value of an user-defined operator over
    a surface of constant-binding energy
    
    Option of summing over degenerate bands (for e.g. fat bands) with degen boolean flag
        
    *args*:

        - **O**: matrix representation of the operator (numpy array len(basis), len(basis) of complex float)
            
        - **TB**: Tight binding object from *chinook.TB_lib.py*
        
        - **ktuple**: momentum range for mesh: 
            ktuple[0] = (x0,xn,n),ktuple[1]=(y0,yn,n),ktuple[2]=kz
            
    *kwargs*:

        - **vlims**: limits for the colourscale (optional argument), will choose 
        a reasonable set of limits if none passed by user
        
        - **ax**: matplotlib Axes, option for plotting onto existing Axes


    *return*:

        - **pts**: the numpy array of expectation values, of shape Nx3, with first
        two dimensions the kx,ky coordinates of the point, and the third the expectation
        value.
        
        - **ax**: matplotlib Axes, allowing for further user modifications
        
    ***
    '''

    
    coords,Eb,Ev=FS(TB,ktuple,Ef,tol)
    masked_Ev = np.array([Ev[int(coords[ei,2]/len(TB.basis)),:,int(coords[ei,2]%len(TB.basis))] for ei in range(len(coords))])
    Ovals = np.sum(np.conj(masked_Ev)*np.dot(O,masked_Ev.T).T,1)

    if np.sign(Ovals.min())!=np.sign(Ovals.max()):
        cmap = cm.RdBu
    else:
        cmap = cm.magma
    
    pts = np.array([[coords[ci,0],coords[ci,1],np.real(Ovals[ci])] for ci in range(len(coords))])
    
    if vlims==(0,0):
        vlims = (Ovals.min()-(Ovals.max()-Ovals.min())/10.0,Ovals.max()+(Ovals.max()-Ovals.min())/10.0)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(pts[:,0],pts[:,1],c=pts[:,2],cmap=cmap,s=200,vmin=vlims[0],vmax=vlims[1])
    ax.scatter(pts[:,0],pts[:,1],c='k',s=5)
    
    return pts,ax



def FS(TB,ktuple,Ef,tol,ax=None):
    
    '''
    A simplified form of Fermi surface extraction, for proper calculation of this,
    *chinook.FS_tetra.py* is preferred. This finds all points in kmesh within a 
    tolerance of the constant energy level.
    
    *args*:

        - **TB**: tight-binding model object
        
        - **ktuple**: tuple of k limits, len (3), First and second should be iterable,
        define the limits and mesh of k for kx,ky, the third is constant, float for kz
        
        - **Ef**: float, energy of interest, eV
        
        - **tol**: float, energy tolerance, float
        
        - **ax**: matplotlib Axes, option for plotting onto existing Axes

    
    *return*:

        - **pts**: numpy array of len(N) x 3 indicating x,y, band index
        
        - **TB.Eband**: numpy array of float, energy spectrum
        
        - **TB.Evec**: numpy array of complex float, eigenvectors
        
        - **ax**: matplotlib Axes, for further user modification
    
    ***
    '''
    
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
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(pts[:,0],pts[:,1])
    return pts,TB.Eband,TB.Evec,ax
    
    
####################SOME STANDARD OPERATORS FOLLOW HERE: ######################
    
    

def LdotS(TB,axis=None):
    '''
    Wrapper for **O_path** for computing L.S along a vector projection of interest,
    or none at all.
    
    
    *args*:

        - **TB**: tight-binding obect
        
    *kwargs*:

        - **axis**: numpy array of 3 float, indicating axis, or None for full L.S
    
    *return*:

        - **O**: numpy array of Nxlen(basis) float, expectation value of operator
        on each band over the kpath of TB.Kobj.
    
    ***
    '''
    HSO = LSmat(TB,axis)
    O = O_path(HSO,TB,TB.Kobj)
    return O

def Sz(TB):
    '''
    Wrapper for **O_path** for computing Sz along a vector projection of interest,
    or none at all.
    
    
    *args*:

        - **TB**: tight-binding obect
        
    
    *return*:

        - **O**: numpy array of Nxlen(basis) float, expectation value of operator
        on each band over the kpath of TB.Kobj.
    
    ***
    '''
    Omat = S_vec(len(TB.basis),np.array([0,0,1]))
    O = O_path(Omat,TB,TB.Kobj)
    return O



def surface_proj(basis,length):
    '''
    Operator for computing surface-projection of eigenstates. User passes the orbital basis
    and an extinction length (1/e) length for the 'projection onto surface'. The operator 
    is diagonal with exponenential suppression based on depth.
    
    For use with SLAB geometry only
    
    *args*:

        - **basis**: list, orbital objects
        
        - **cutoff**: float, cutoff length
    
    *return*:

        - **M**: numpy array of float, shape len(TB.basis) x len(TB.basis)
        
    ***
    '''
    Omat = np.identity(len(basis))
    attenuation = np.exp(np.array([-abs(o.depth)/length for o in basis]))
    return Omat*attenuation
    

def S_vec(LB,vec):
    '''
    Spin operator along an arbitrary direction can be written as
    n.S = nx Sx + ny Sy + nz Sz
    
    *args*:

        - **LB**: int, length of basis
        
        - **vec**: numpy array of 3 float, direction of spin projection
    
    *return*:

        - numpy array of complex float (LB by LB), spin operator matrix
    
    ***
    '''
    numstates = int(LB/2)
    Si = 0.5*np.identity(numstates,dtype=complex)
    
    Smats = np.zeros((3,LB,LB),dtype=complex)
    Smats[2,:numstates,:numstates]+=-Si
    Smats[2,numstates:,numstates:]+=Si
    Smats[0,:numstates,numstates:] +=Si
    Smats[0,numstates:,:numstates]+=Si
    Smats[1,:numstates,numstates:]+=1.0j*Si
    Smats[1,numstates:,:numstates]+=-1.0j*Si
    
    return vec[0]*Smats[0]+vec[1]*Smats[1]+vec[2]*Smats[2]

    


def is_numeric(a):

    '''
    Quick check if object is numeric
    
    *args*:

        - **a**: numeric, float/int
    
    *return*:

        - bool, if numeric True, else False

    ***
    '''
    if a is not None:
        
        try:
            float(a)
            return True
        except ValueError:
            return False
    else:
        return False
    
    