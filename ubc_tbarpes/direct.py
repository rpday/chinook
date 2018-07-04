# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 18:06:00 2018

@author: rday

Explicit integration of the wavefunctions in graphite

"""
import sys
sys.path.append('/Users/ryanday/Documents/UBC/TB_python/TB_ARPES-rpday-patch-2/')


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import ubc_tbarpes.Ylm as Ylm
import ubc_tbarpes.electron_configs as econ
import ubc_tbarpes.klib as klib
from operator import itemgetter

hb = 6.626*10**-34/(2*np.pi)
c  = 3.0*10**8
q = 1.602*10**-19
A = 10.0**-10
me = 9.11*10**-31
mN = 1.67*10**-27
kb = 1.38*10**-23



######---------------------------PSI PLOTTING----------------------------######
def gen_map(orbital,X,Y,Z):
    '''
    Tool for imaging the eigenstate, projected onto the lattice orbital basis
    args:
        orbital -- instance of orbital object from the ubc_tbarpes library
        X,Y,Z -- mesh of real-space points at which to compute the value of the wavefunction
    return:
        state: numpy array of complex float, of same size as X,Y,Z corresponding to wavefunction value at all points
    '''
    
    R = np.sqrt(abs(abs(X-orbital.pos[0])**2+abs(Y-orbital.pos[1])**2+abs(Z-orbital.pos[2])**2))
    orb,_,_ = econ.Slater(orbital.Z,orbital.label,R)
    TH = np.arccos((Z-orbital.pos[2])/R)
    PH = np.arctan2((Y-orbital.pos[1]),(X-orbital.pos[0]))
    state = np.zeros(np.shape(X),dtype=complex)
    for op in orbital.proj:
        state+=orb*Ylm.Y(op[2],op[3],TH,PH)*(op[0]+op[1]*1.0j)
    return state




def plt_map(psi,x,y,z,inds):
    '''
    Tool for plotting the wavefunction as an intensity map, cut through the cartesian planes.
    args:
        psi -- wavefunction, as computed by gen_map, over the range of points given by 
        x,y,z -- numpy arrays of float
        inds -- indices wanted to plot, int
    return None
    '''
    Xxy,Yxy = np.meshgrid(x,y)
    Xxz,Yxz = np.meshgrid(x,z)
    Xyz,Yyz = np.meshgrid(y,z)
    fig = plt.figure()
    ax = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    p = ax3.pcolormesh(Xxy,Yxy,psi[:,:,inds[2]],cmap=cm.RdBu,vmin = -abs(psi).max(),vmax=abs(psi).max())
    p2 = ax2.pcolormesh(Xxz,Yxz,psi[:,inds[1],:],cmap=cm.RdBu,vmin = -abs(psi).max(),vmax=abs(psi).max())
    p3 = ax.pcolormesh(Xyz,Yyz,psi[inds[0],:,:],cmap=cm.RdBu,vmin = -abs(psi).max(),vmax=abs(psi).max())
    
    
def plt_3d(X,Y,Z,P,tol):
    '''
    Tool for plotting wavefunction as array of points in R3, as computed with gen_map.
    args:
        X,Y,Z -- numpy arrays of points forming same meshgrid as used in gen_map
        P -- numpy array of complex float (wavefunction from gen_map)
        tol -- tolerance for plotting (to avoid excessive memory consumption)
    '''
    PF = P.flatten()
    pts = np.array([[X.flatten()[i],Y.flatten()[i],Z.flatten()[i],P.flatten()[i]] for i in range(len(PF)) if abs(PF[i])>tol])
    sz = 50*abs(pts[:,3])
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(pts[:,0],pts[:,1],pts[:,2],s=sz,c=np.real(pts[:,3]),cmap=cm.RdBu,vmin=-abs(pts[:,3]).max()/3,vmax=abs(pts[:,3]).max()/3)
######--------------------------PSI PLOTTING END-------------------------######


######-------------------Direct Transition Operator----------------------######

def dir_mat(basis,origin,rx,N):
    ''' 
    Generate an operator matrix for the direct transition, given light polarization
    along each of the three cardinal axes. This is then a (len(basis)) x (len(basis)) x 3 numpy array
    '''
    x,y,z=np.linspace(-rx+origin[0],rx+origin[0],N),np.linspace(-rx+origin[1],rx+origin[1],N),np.linspace(-rx+origin[2],rx+origin[2],N)
    X,Y,Z= np.meshgrid(x,y,z)   
    d3r = (x[-1]-x[0])/(len(x))*(y[-1]-y[0])/(len(y))*(z[-1]-z[0])/(len(z))  
    psi_funcs = [gen_map(o,X,Y,Z) for o in basis]
    
    
    dM = np.zeros((len(basis),len(basis),3),dtype=complex)
    for i in range(len(basis)):
        for j in range(i,len(basis)):
            dM[i,j,:] = dir_el(psi_funcs[i],psi_funcs[j],X,Y,Z,d3r)
            if i!=j:
                dM[j,i,:] = np.conj(dM[i,j,:])
    return dM
    
    
def dir_el(psi1,psi2,X,Y,Z,d3r):
    '''
    Generate matrix element for the direct transition matrix
    args:
        psi1,psi2 -- numpy array of complex of same size as X,Y,Z--sampling of orbital in R3
        X,Y,Z -- numpy array of float, sampling range of orbital
        d3r -- float, volume element for renormalizing the finite sum to an effective integral
    '''
 
    mapz = Z*np.conj(psi2)*psi1
    mapx = X*np.conj(psi2)*psi1
    mapy = Y*np.conj(psi2)*psi1
    return np.array([d3r*np.sum(mapx),d3r*np.sum(mapy),d3r*np.sum(mapz)])



def path_dir_op(TB,Kobj,hv,width,T,pol=None):
    '''
    Compute the direct transition probability for a set of bands, over a path in k-space.
    args: 
        TB -- ubc_tbarpes Tight-binding model object, with Energy eigenvalues and Eigenvectors already calculated. If not, solve the model Hamiltonian
        Kobj-- ubc_tbarpes momentum object (K-object) 
        hv -- photon energy (float) in units of eV
        width -- float width of excitation (modeled as a Lorentzian)
        pol -- polarization vector numpy array of length 3 complex float(optional kwarg) 
    return: 
        dip_jdos -- dipole-permitted joint-density of states (which is effectively what we're calculating), numpy array of float with shape len(k)xlen(basis)x3 for 3 Cartesian polarization basis vectors
    '''
    try:
        klen,blen = np.shape(TB.Eband)
    except ValueError:
        TB.solve_H()
        klen,blen = np.shape(TB.Eband)
    dip_jdos = np.zeros((klen,blen,3),dtype=complex)
    thermal = vf(TB.Eband/(kb*T/q))
    dmat = dir_mat(TB.basis,*(np.array([1e-9,1e-9,1e-9]),11,65))
    for i in range(1,klen):
        for ei in range(blen):
            for ej in range(ei):
                if (-3*width)<=((TB.Eband[i,ei]-TB.Eband[i,ej])-hv)<(3*width):
                    line = (thermal[i,ei]-thermal[i,ej])*lorentzian(width,(TB.Eband[i,ei]-TB.Eband[i,ej]),hv)
                    xtrans = np.dot(np.conj(TB.Evec[i][:,ei]),np.dot(dmat[:,:,0],TB.Evec[i][:,ej]))
                    ytrans = np.dot(np.conj(TB.Evec[i][:,ei]),np.dot(dmat[:,:,1],TB.Evec[i][:,ej]))
                    ztrans = np.dot(np.conj(TB.Evec[i][:,ei]),np.dot(dmat[:,:,2],TB.Evec[i][:,ej]))
                    dip_jdos[i,ei,:] += np.array([xtrans,ytrans,ztrans])*line
    if pol is not None:
        plot_jdos(Kobj,TB,dip_jdos,pol)
    return dip_jdos




######-----------------Direct Transition Operator END--------------------######

def optical_conductivity(TB,avec,N,T,dE,pol=None):
    '''
    Compute optical conductivity of sample, over a mesh covering the Brillouin zone.
    This is fairly preliminary, does not perform any interpolation, which is an essential next step.
    args:
        TB  -- instance of the TB class from ubc_tbarpes library
        avec -- numpy array of size 3x3 of float corresponding to row matrix of lattice vectors
        N -- mesh density, the BZ will have ~(2N+1)^3 points
        T -- temperature
    '''
    TB.Kobj = klib.kpath(klib.b_zone(avec,N))
    _,_ = TB.solve_H()
    sig = []
    thermal = vf(TB.Eband/(kb*T/q))
    dmat = dir_mat(TB.basis,*(np.array([1e-9,1e-9,1e-9]),11,65))
    klen = len(TB.Kobj.kpts)
    blen = len(TB.basis)
    fer = []
    for i in range(1,klen):
        for ei in range(blen):
            for ej in range(ei):
                dE = TB.Eband[i][ei] - TB.Eband[i][ej]
                line = (thermal[i,ej]-thermal[i,ei]) #occupation of occupied - occupation of unoccupied
                if line>0:
                    xtrans = np.dot(np.conj(TB.Evec[i][:,ei]),np.dot(dmat[:,:,0],TB.Evec[i][:,ej]))
                    ytrans = np.dot(np.conj(TB.Evec[i][:,ei]),np.dot(dmat[:,:,1],TB.Evec[i][:,ej]))
                    ztrans = np.dot(np.conj(TB.Evec[i][:,ei]),np.dot(dmat[:,:,2],TB.Evec[i][:,ej]))
                    sig.append([dE,line*abs(xtrans)**2,line*abs(ytrans)**2,line*abs(ztrans)**2])
                    fer.append([dE,line])
                
    
    sig = np.array(sorted(sig,key=itemgetter(0)))
    e,x = resample(sig[:,0],sig[:,1],dE)
    _,y = resample(sig[:,0],sig[:,2],dE)
    _,z = resample(sig[:,0],sig[:,3],dE)
    sig = np.array([e,x,y,z]).T
    fig = plt.figure()
    ax = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    ax.plot(sig[:,0],sig[:,1])
    ax2.plot(sig[:,0],sig[:,2])
    ax3.plot(sig[:,0],sig[:,3])
    
    
    return sig


def resample(x,y,dx):
    xp = np.arange(x.min(),x.max(),dx)
    yp = np.zeros(len(xp))
    for i in range(len(x)):
        ind = int((x[i]-xp[0])/dx)
        yp[ind] +=y[i]
    return xp,yp
    
    
    


######-------------Direct Transition SLOWWWWW VERSION--------------------######    

def build_state(psi,X,Y,Z,basis):
    state = np.zeros(np.shape(X),dtype=complex)
    for pi in range(len(psi)):
        state+=psi[pi]*gen_map(basis[pi],X,Y,Z)
    return state



def dir_trans(psi_vec,basis,origin,rx,N):
    '''
    Transition probabilty (matrix element) between two states psi_vec[0], psi_vec[1]
    '''
    x,y,z=np.linspace(-rx+origin[0],rx+origin[0],N),np.linspace(-rx+origin[1],rx+origin[1],N),np.linspace(-rx+origin[2],rx+origin[2],N)
    X,Y,Z= np.meshgrid(x,y,z)
    psi1 = build_state(psi_vec[0],X,Y,Z,basis)
    psi2 = build_state(psi_vec[1],X,Y,Z,basis)
    
    d3r = (x[-1]-x[0])/(len(x))*(y[-1]-y[0])/(len(y))*(z[-1]-z[0])/(len(z))    
    mapz = Z*np.conj(psi2)*psi1
    mapx = X*np.conj(psi2)*psi1
    mapy = Y*np.conj(psi2)*psi1
    M = np.array([d3r*np.sum(mapx),d3r*np.sum(mapy),d3r*np.sum(mapz)])
    return M

def lorentzian(width,e,pk):
    '''
    Normalized Lorentzian function with peak at pk and width of width. 
    args:
        width -- width of Lorentzian
        e -- numpy array of points over which to evaluate the Lorentzian
        pk -- peak position of the function
    return:
        numpy array of Lorentzian function with above parameters, over the domain of e
    '''
    return width/(2)/((e-pk)**2+(width/2)**2)

def path_direct(TB,Kobj,hv,width,T,pol=None):
    '''
    Compute the direct transition probability for a set of bands, over a path in k-space.
    args: 
        TB -- ubc_tbarpes Tight-binding model object, with Energy eigenvalues and Eigenvectors already calculated. If not, solve the model Hamiltonian
        Kobj-- ubc_tbarpes momentum object (K-object) 
        hv -- photon energy (float) in units of eV
        width -- float width of excitation (modeled as a Lorentzian)
        pol -- polarization vector numpy array of length 3 complex float(optional kwarg) 
    return: 
        dip_jdos -- dipole-permitted joint-density of states (which is effectively what we're calculating), numpy array of float with shape len(k)xlen(basis)x3 for 3 Cartesian polarization basis vectors
    '''
    try:
        irng,jrng = np.shape(TB.Eband)
    except ValueError:
        TB.solve_H()
        irng,jrng = np.shape(TB.Eband)
    dip_jdos = np.zeros((irng,jrng,3),dtype=complex)
    thermal = vf(TB.Eband/(kb*T/q))
    for i in range(1,irng):
        for ei in range(jrng):
            for ej in range(ei):
                if (-3*width)<=((TB.Eband[i,ei]-TB.Eband[i,ej])-hv)<(3*width):
                    dip_jdos[i,ei,:] += dir_trans((TB.Evec[i][:,ei],TB.Evec[i][:,ej]),TB.basis,np.array([1e-6,1e-6,1e-6]),11,65)*(thermal[i,ei]-thermal[i,ej])*lorentzian(width,(TB.Eband[i,ei]-TB.Eband[i,ej]),hv)
    if pol is not None:
        plot_jdos(Kobj,TB,dip_jdos,pol)
    return dip_jdos

def plot_jdos(Kobj,TB,dip_jdos,pol):
    '''
    Plot the joint density of states as calculated in path_direct function.
    args:
        TB -- ubc_tbarpes Tight-binding model object, with Energy eigenvalues and Eigenvectors already calculated. If not, solve the model Hamiltonian
        Kobj -- ubc_tbarpes momentum object (K-object) 
        dip_jdos -- dipole permitted joint density of states (as computed by path_direct), numpy float of size len(K)xlen(TB.basis)x3
        pol -- numpy float: polarization vector for excitation light
    
    return:
        jdos: numpy float size len(k)x len(TB.basis) 
    '''
    jdos = abs(np.dot(dip_jdos,pol))**2
    fig = plt.figure()
    
    ax = fig.add_subplot(111)
    for ei in range(np.shape(TB.Eband)[1]):
        plt.plot(Kobj.kcut,TB.Eband[:,ei],c='grey')
        ax.scatter(Kobj.kcut,TB.Eband[:,ei],c=jdos[:,ei],cmap=cm.PuBu)
    
    return jdos

def k_integrated(Kobj,TB,width,Ij,ylim=None):
    dE = abs(np.array([[TB.Eband[i+1,j]-TB.Eband[i,j] for i in range(len(Kobj.kpts)-1)] for j in range(np.shape(TB.Eband)[1])])).mean()
    en_dig = np.arange(TB.Eband.min()-0.5,TB.Eband.max()+0.5,dE)
    Im = np.zeros((len(Kobj.kpts),len(en_dig)))
    for i in range(np.shape(TB.Eband)[1]):
        for j in range(np.shape(TB.Eband)[0]):
            Im[j,:]+=lorentzian(width,en_dig,TB.Eband[j,i])*Ij[j,i]
    Ijdos = np.sum(Im,0)
    fig = plt.figure()
    
    K,E = np.meshgrid(Kobj.kpts[:,1],en_dig)
    ax = fig.add_subplot(121)
    ax.pcolormesh(K,E,Im.T,cmap=cm.Spectral,vmin = 0,vmax=abs(Im.max()/1))
    for bn in range(np.shape(TB.Eband)[1]):
        ax.plot(Kobj.kpts[:,1],TB.Eband[:,bn],lw=0.5,c='k')
    ax2 = fig.add_subplot(122)
    plt.plot(Ijdos,en_dig)
    if ylim is not None:
        ax.set_ylim(ylim[0],ylim[1])
        ax2.set_ylim(ylim[0],ylim[1])
    return Im,Ijdos





def con_ferm(x):      ##Typical energy scales involved and temperatures involved give overflow in exponential--define a new fermi function that works around this problem
    tmp = 0.0
    if x<709:
        tmp = 1.0/(np.exp(x)+1)
    return tmp


vf = np.vectorize(con_ferm)
        
        

  
#if __name__ == "__main__":
#
#    a,c = 2.46,3.35
#    centres = np.array([np.array([1e-10,1e-10,1e-10]),np.array([-a/np.sqrt(3.0),0.0,0.0]),np.array([0,0,c]),np.array([-a/np.sqrt(12),a/2,c])])
#    origin = np.sum(centres,0)/4 #0.5*(centres[1])#np.zeros(3)#
#    psi_1 = np.array([np.sqrt(0.5),0,0,-np.sqrt(0.5)])
#    psi_4 = np.array([0,np.sqrt(0.5),-np.sqrt(0.5),0])
#    
#    rx = 15
#    Ni = []
#    Ix,Iy,Iz=[],[],[]
#    Npts = 70
#    for i in range(2):
#        Npts = 50+i*50
#        x = np.linspace(-rx+origin[0],rx+origin[0],Npts)
#        y = np.linspace(-rx+origin[1],rx+origin[1],Npts)
#        z = np.linspace(-rx+origin[2],rx+origin[2],Npts)
#        X,Y,Z = np.meshgrid(x,y,z)
#    #    psi1 = np.sqrt(0.5)*(gen_map(centres[0],X,Y,Z)+gen_map(centres[1],X,Y,Z))
#    #    psi2 = np.sqrt(0.5)*(gen_map(centres[0],X,Y,Z)-gen_map(centres[1],X,Y,Z))
#    #    plt_3d(X,Y,Z,psi1,0.005)
#        
#    ##K POINT
#        psi1 = np.sqrt(0.5)*(gen_map(centres[0],X,Y,Z)-gen_map(centres[2],X,Y,Z))
#        psi4 = np.sqrt(0.5)*(gen_map(centres[0],X,Y,Z)+gen_map(centres[2],X,Y,Z))
#        psi2 = np.sqrt(0.5)*(gen_map(centres[1],X,Y,Z)+gen_map(centres[3],X,Y,Z))
#        psi3 = np.sqrt(0.5)*(gen_map(centres[1],X,Y,Z)-gen_map(centres[3],X,Y,Z))
#    ##0.895 K
#        p1 = np.array([ 0.5402+0.5j, -0.4563-0.0, -0.5402-0.0,  0.4563+0.0])
#        p2 = np.array([-0.437 -0.0,  0.5559+0.0, -0.437 -0.0,  0.5559+0.0])
#        p3 = np.array([ 0.4563+0.0,  0.5402+0.0, -0.4563-0.0, -0.5402+0.0])
#        p4 = np.array([-0.5559-0.0, -0.437 -0.0, -0.5559-0.0, -0.437 -0.0])
#        
#        
#        
#        psi1 = build_state(p1,X,Y,Z,centres)
#        psi2 = build_state(p2,X,Y,Z,centres)
#        psi3 = build_state(p3,X,Y,Z,centres)
#        psi4 = build_state(p4,X,Y,Z,centres)
#        
#        
#        mapz = Z*psi2*psi3
#        mapx = X*psi2*psi3
#        mapy = Y*psi2*psi3
#        dx,dy,dz = (x[-1]-x[0])/(len(x)),(y[-1]-y[0])/(len(y)),(z[-1]-z[0])/(len(z))
#        d3r = dx*dy*dz
#        xint,yint,zint = d3r*np.sum(mapx),d3r*np.sum(mapy),d3r*np.sum(mapz)
#        print(xint,yint,zint)
#        Ni.append(Npts)
#        Ix.append(xint)
#        Iy.append(yint)
#        Iz.append(zint)
##        
##    fig = plt.figure()
##    plt.plot(Ni,Ix)
##    fig = plt.figure()
##
##    plt.plot(Ni,Iy)
##    fig = plt.figure()
##
##    plt.plot(Ni,Iz)
##         
##    plt_3d(X,Y,Z,psi1,0.005)
##    plt_3d(X,Y,Z,psi2,0.005)
##    plt_3d(X,Y,Z,psi3,0.005)
##    plt_3d(X,Y,Z,psi4,0.005)
###    plt_map(psi1,x,y,z,[100,100,100])
##    plt_map(psi2,x,y,z,[100,100,100])
##    plt_map(mapx,x,y,z,[100,100,100])
##    plt_map(mapy,x,y,z,[100,100,100])
##    plt_map(mapz,x,y,z,[100,100,100])