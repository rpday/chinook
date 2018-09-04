# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 16:02:49 2018

@author: rday

Calculation of Density of States

"""


import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
import ubc_tbarpes.klib as klib

class dos_env:
    
    def __init__(self,TB):
        self.TB = TB
        self.pdos = []
        
    def do_dos(self,N):
        self.TB.Kobj.kpts = self.build_bz(N)
        self.Evals,self.Evecs,self.Nbins = self.prep_bands()
        self.hi = self.solve_dos()
                
        
    def build_bz(self,N):
        kpts = klib.b_zone(self.TB.avec,N)
        print(len(kpts))
        return kpts
    
    def prep_bands(self):
        self.TB.solve_H()
        Evals,Evecs = sort_bands(self.TB.Eband,self.TB.Evec)
        Erange = (Evals.min(),Evals.max())
        dE = np.array([Evals[i+1]-Evals[i] for i in range(len(Evals)-1)]).max()    
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
        dos = hi[0].astype(float)
#        rescale = dos.max()
#        dos*=1./rescale
        
    
        print('Density of States calculation complete.')
    
        return (dos,hi[1])
        
        
    def calc_pdos(self,projs):

        orb = pdos_project(self.TB.basis,projs)
        weights = abs(np.dot(self.Evecs,orb))**2
        dos_proj = np.histogram(self.Evals,bins=self.Nbins,weights=weights)
        print('Partial Density of States calculation complete.')
#        dos_proj[0][:]*=1./rescale
        return dos_proj
        
    
#def dos_prep(TB,N,evecs=False):
#    '''
#    Prepare bands and eigenvectors for Density of States of a tight-binding model.
#    User defines a choice of k-mesh via N, which can be passed as either:
#        integer or list/tuple/array of 3 integers
#    The bandstructure is binned corresponding to the average energy-level spacing
#    and then plotted 
#    '''
#    print('Building Brillouin Zone k-mesh...')
#    blatt = _b_lattice_(TB.avec)
#    BZ = BZ_gen(blatt,raw_mesh(blatt,N))
#    TB.Kobj.kpts = BZ
#    print('Total k-mesh points: {:d}. Diagonalizing Hamiltonian over the Brillouin zone...'.format(len(BZ)))
#    TB.solve_H()
#    print('Diagonalization Complete! Computing Density of States...')
#    if evecs==False:
#        Evals = sort_bands(TB.Eband)
#        return Evals
#    else:
#        Evals,Evecs = sort_bands(TB.Eband,TB.Evec)
#        return Evals,Evecs




    
def dos_calc(TB,N,pdos=False,pind=0):
    '''
    Calculate the density of states, given a mesh scale over the Brillouin zone.
    Option to perform a partial-density of states instead. Want to modify this in the
    near future so that the DOS need not be re-calculated every time you choose a different projection.
    args:
        TB -- instance of tight-binding model object
        N -- mesh scaling, integer or list/tuple/array of 3 integers
        pdos -- Bool
    '''
    if pdos == False:
        Evals = dos_prep(TB,N)
    else:
        Evals,Evecs = dos_prep(TB,N,True)
        orb = pdos_project(TB.basis,pind)
        weights = abs(np.dot(Evecs,orb))**2
    
    Erange = (Evals.min(),Evals.max())
    dE = np.array([Evals[i+1]-Evals[i] for i in range(len(Evals)-1)]).max()    
    Nbins = int((Erange[1]-Erange[0])/dE)
    hi = np.histogram(Evals,bins=Nbins)
    if pdos:
        dos_proj = np.histogram(Evals,bins=Nbins,weights=weights)
        print('Density of States calculation complete.')

        return hi,dos_proj
    else:
        print('Density of States calculation complete.')

        return hi



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
    return ovec
    
                
            
        
    



#if __name__ == "__main__":
#    TB = FeSe.build_TB()
#    EB,ED = dos_calc(TB,(10,10,5))
#    
#    

    

