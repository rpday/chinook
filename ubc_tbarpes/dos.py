# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 16:02:49 2018

@author: rday

Calculation of Density of States

"""


import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
import scipy.ndimage as nd
import ubc_tbarpes.klib as klib
import ubc_tbarpes.tetrahedra as tetrahedra

class dos_env:
    
    def __init__(self,TB,sigma=0.0):
        self.TB = TB
        self.pdos = []
        self.sigma = sigma
        
    def do_dos(self,N):
        self.TB.Kobj.kpts = self.build_bz(N)
        self.Evals,self.Evecs,self.Nbins = self.prep_bands()
        self.hi = self.solve_dos()

        
                
        
    def build_bz(self,N):
        if type(N)==int:
            N = (N,N,N)
        
        b_vec = klib.bvectors(self.TB.avec)
        x,y,z = np.linspace(0,1,N[0]),np.linspace(0,1,N[1]),np.linspace(0,1,N[2])
        X,Y,Z = np.meshgrid(x,y,z)
        X,Y,Z = X.flatten(),Y.flatten(),Z.flatten()
        kpts = np.dot(np.array([[X[i],Y[i],Z[i]] for i in range(len(X))]),b_vec)
#        kpts = klib.b_zone(self.TB.avec,N)
        print(len(kpts))
        return kpts
    
    def prep_bands(self):
        self.TB.solve_H()
        Evals,Evecs = sort_bands(self.TB.Eband,self.TB.Evec)
        Erange = (Evals.min(),Evals.max())
        Ediff = abs(np.array([Evals[i+1]-Evals[i] for i in range(len(Evals)-1)]))
        dE = (Ediff[Ediff>0]).mean()*30#(Ediff.min()+Ediff.max())/2
        Nbins = int((Erange[1]-Erange[0])/dE)
        return Evals,Evecs,Nbins
    
    def solve_dos(self):
        '''
        Calculate the density of states, given a mesh scale over the Brillouin zone.
        Option to perform a partial-density of states instead. Want to modify this in the
        near future so that the DOS need not be re-calculated every time you choose a different projection.
        args:
            TB -- instance of tight-binding model object
            N -- mesh scaling, integer or list/tuple/array of 3 integers
            pdos -- Bool
        '''
        
        hi = np.histogram(self.Evals,bins=self.Nbins)
        de = hi[1][1]-hi[1][0]
        if abs(self.sigma)>0:
            dos = nd.gaussian_filter1d(hi[0].astype(float),self.sigma/de)
        else:
            dos = hi[0]
#        rescale = dos.max()
#        dos*=1./rescale
        
    
        print('Density of States calculation complete.')
    
        return (dos,hi[1])
        
        
    def calc_pdos(self,projs):

        weights = np.sum(abs(self.Evecs[:,projs])**2,1)
        dos_proj = np.histogram(self.Evals,bins=self.Nbins,weights=weights)
        de = self.hi[1][1]-self.hi[1][0]
        if abs(self.sigma)>0:
            pdos = nd.gaussian_filter1d(dos_proj[0].astype(float),self.sigma/de)
        print('Partial Density of States calculation complete.')
#        dos_proj[0][:]*=1./rescale
        return (pdos,dos_proj[1])
        



def plot_dos(hi,pdos=None):
    dosvals = (hi[0]).astype(float)
    dosmax = dosvals.max()
    dosvals = (dosvals)/dosmax

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(hi[1][:-1],dosvals)
    
    if pdos is not None:
        pdos_scale = pdos/dosmax
        ax.plot(hi[1][:-1],pdos_scale)
    
    
    ax.axis([hi[1][0],hi[1][-1],0,1.1])
    ax.set_xlabel('Energy (eV)')
    ax.set_ylabel('Density of States (a.u.)')

def sort_bands(bands,vecs=None):
    if vecs is None:
        Evals = np.array(sorted(bands.flatten()))
        return Evals
    else:
        Ebands = bands.flatten()
        Evecs = np.array([vecs[k,:,j] for k in range(np.shape(bands)[0]) for j in range(np.shape(bands)[1])])
        Eb_v = np.zeros((len(Ebands),np.shape(bands)[1]+1),dtype=complex)
        Eb_v[:,0] = Ebands
        Eb_v[:,1:] = Evecs
        Eb_v = np.array(sorted(Eb_v,key=itemgetter(0)))
        return np.real(Eb_v[:,0]),Eb_v[:,1:]
        



def pdos_project(basis,inds):
  #  orb_in = input('Enter the orbital projection of interest (integer indices)')
    ovec = np.zeros(len(basis))
    
    ovec[inds] = 1
#    ovec/=np.linalg.norm(ovec)
    return ovec


################# Density of States following the Blochl Prescription #######################
###############https://journals.aps.org/prb/pdf/10.1103/PhysRevB.49.16223####################
    
def dos_tetra(TB,NE,NK,kzval=None):
    '''
    Generate a tetrahedra mesh of k-points which span the BZ with even distribution
    Diagonalize over this mesh and then compute the resulting density of states as
    prescribed in the above paper. 
    The result is plotted, and DOS returned
    args:
        TB -- tight-binding model object
        NE -- integer, number of energy points
        NK -- integer / list of 3 integers -- number of k-points in mesh
    return:
        Elin -- linear energy array of float, spanning the range of the eigenspectrum
        DOS -- density of states numpy array of float, same length as Elin
    '''
    print('tetrahedra.mesh modified for a "fixed" kz value!')
 #   kpts,tetra = tetrahedra.mesh_tetra(TB.avec,NK,kzval) ####EDITED FOR GRAPHITE!!!!!!!#####
    kpts,tetra = tetrahedra.mesh_tetra_dos(TB.avec,NK)
   # print('kzvals:',set(kpts[:,2]))
    TB.Kobj.kpts = kpts
    TB.solve_H()
    Elin = np.linspace(TB.Eband.min(),TB.Eband.max(),NE)

    DOS = np.zeros(len(Elin))
    for ki in range(len(tetra)):
        E_tmp = TB.Eband[tetra[ki]]
        for bi in range(len(TB.basis)): #iterate over all bands
            Eband = sorted(E_tmp[:,bi])
            args = (*Eband,1,len(tetra))
            DOS += dos_func(Elin,args)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(Elin,DOS)               
    return Elin,DOS,TB
        
    
##############################-------D(E)---------#############################
def dos_func(e,epars):
    '''
    Piecewise function for calculation of density of states
    args:
        e -- numpy array of float (energy domain)
        epars -- tuple of parameters: e[0],e[1],e[2],e[3],V_T,V_G being the ranked band energies for the tetrahedron, 
        as well as the volume of both the tetrahedron and the Brillouin zone, all float
    return:
        numpy array of float giving DOS contribution from this tetrahedron
    '''
    return np.piecewise(e,[e<epars[0],(epars[0]<=e)*(e<epars[1]),(epars[1]<=e)*(e<epars[2]),(epars[2]<=e)*(e<epars[3]),e>=epars[3]],[e_out,e_12,e_23,e_34,e_out],epars)


def e_out(e,epars):
    return np.zeros(len(e))

def e_12(e,epars):
    return epars[4]/epars[5]*3*(e-epars[0])**2/(epars[1]-epars[0])/(epars[2]-epars[0])/(epars[3]-epars[0])

def e_23(e,epars):
    e21,e31,e41,e42,e32 = epars[1]-epars[0],epars[2]-epars[0],epars[3]-epars[0],epars[3]-epars[1],epars[2]-epars[1]
    e2 = e-epars[1]
    return epars[4]/epars[5]/e31/e41*(3*e21+6*e2-3*(e31+e42)/e32/e42*e2**2)

def e_34(e,epars):
    return epars[4]/epars[5]*3*(epars[3]-e)**2/(epars[3]-epars[0])/(epars[3]-epars[1])/(epars[3]-epars[2])
##############################-------D(E)---------#############################
    

##############################-------n(E)---------#############################
def EF_find(TB,occ,dE,NK):
    '''
    Use the tetrahedron-integration method to establish the Fermi-level, for a given
    electron configuration. 
    args:
        TB -- instance of Tight-Binding model object from TB_lib
        occ -- desired electronic occupation
        dE -- estimate of energy precision you want to evaluate the Fermi-level to (in eV)
        NK -- depth of the k-mesh, as integer or list/tuple
    return:
        e -- numpy array of float: energy array
    '''
    e,n = n_tetra(TB,dE,NK)
    EF = e[np.where(abs(n-occ)==abs(n-occ).min())[0][0]]
    return EF


def n_tetra(TB,dE,NK):
    '''
    This function, also from the algorithm of Blochl, gives the integrated DOS
    at every given energy (so from bottom of bandstructure up to its top. This makes
    for very convenient and precise evaluation of the Fermi level, given an electron
    number)
    args:
        TB -- tight-binding model object
        dE -- float, energy spacing (meV)
        NK -- integer / list of 3 integers -- number of k-points in mesh
    return:
        Elin -- linear energy array of float, spanning the range of the eigenspectrum
        DOS -- density of states numpy array of float, same length as Elin
    '''
    kpts,tetra = tetrahedra.mesh_tetra(TB.avec,NK)
    TB.Kobj.kpts = kpts
    TB.solve_H()
    Elin = np.arange(TB.Eband.min(),TB.Eband.max(),dE)
    n = np.zeros(len(Elin))
    for ki in range(len(tetra)):
        E_tmp = TB.Eband[tetra[ki]]
        for bi in range(len(TB.basis)): #iterate over all bands
            Eband = sorted(E_tmp[:,bi])
            args = (*Eband,1,len(tetra))
            n += n_func(Elin,args)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(Elin,n)               
    return Elin,n    



def n_func(e,epars):
    
    
    return np.piecewise(e,[e<epars[0],(epars[0]<=e)*(e<epars[1]),(epars[1]<=e)*(e<epars[2]),(epars[2]<=e)*(e<epars[3]),e>=epars[3]],[n1,n12,n23,n34,n4],epars)

def n1(e,epars):
    return np.zeros(len(e))

def n12(e,epars):
    return epars[4]/epars[5]*(e-epars[0])**3/(epars[1]-epars[0])/(epars[2]-epars[0])/(epars[3]-epars[0])

def n23(e,epars):
    e21,e31,e41,e42,e32 = epars[1]-epars[0],epars[2]-epars[0],epars[3]-epars[0],epars[3]-epars[1],epars[2]-epars[1]
    e2 = e-epars[1]
    return epars[4]/epars[5]*(1/(e31*e41))*(e21**2+3*e21*(e2)+3*e2**2-(e31+e42)/(e32*e42)*(e2**3))

def n34(e,epars):
    return epars[4]/epars[5]*(1-(epars[3]-e)**3/(epars[3]-epars[0])/(epars[3]-epars[1])/(epars[3]-epars[2]))

def n4(e,epars):
    return epars[4]/epars[5]
##############################-------n(E)---------#############################
    

    
if __name__ == "__main__":
    
#    e,n = n_tetra(TB,100,50)
    e,d = dos_tetra(TB,150,18)
    print(np.sum(d*(e[1]-e[0])))
#    TB = FeSe.build_TB()
#    Nit = []
#    Eit = []
#    for i in range(5):
#        En,DOS = dos_tetra(TB,100,40+5*i)
#        Nit.append([40+2*i,np.sum(DOS*(En[1]-En[0]))])
#    for i in range(5):
#        En,DOS = dos_tetra(TB,80+20*i,45)
#        Eit.append([80+10*i,np.sum(DOS*(En[1]-En[0]))])
#    Nit = np.array(Nit)
#    Eit = np.array(Eit)
#    
#    fig = plt.figure()
#    plt.plot(Nit[:,0],Nit[:,1])
#    fig = plt.figure()
#    plt.plot(Eit[:,0],Eit[:,1])
#    
##    
#    

    

