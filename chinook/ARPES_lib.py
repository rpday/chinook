#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Created on Sat Nov 18 21:15:20 2017

#@author: rday

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
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.



import numpy as np

import sys
import datetime as dt

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from scipy.interpolate import interp1d
import scipy.ndimage as nd
from scipy.signal import hilbert

from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

import chinook.klib as K_lib
import chinook.orbital as olib
import chinook.radint_lib as radint_lib

import chinook.Tk_plot as Tk_plot

if Tk_plot.tk_query():
    tk_found = True
else:
    tk_found = False
import chinook.Ylm as Ylm 
import chinook.rotation_lib as rotlib
import chinook.intensity_map as imap
import chinook.tilt as tilt



####PHYSICAL CONSTANTS RELEVANT TO CALCULATION#######
hb = 6.626*10**-34/(2*np.pi)
c  = 3.0*10**8
q = 1.602*10**-19
A = 10.0**-10
me = 9.11*10**-31
mN = 1.67*10**-27
kb = 1.38*10**-23




###
class experiment:
    '''
    The experiment object is at the centre of the ARPES matrix element 
    calculation.This object keeps track of the experimental geometry as 
    well as a local copy of the tight-binding model and its dependents. 
    Such a copy is used to avoid corruption of these objects in the global
    space during a given run of the ARPES experiment.
    
    *args*: 
        
        - **TB**: instance of a tight-binding model object
        
        - **ARPES_dict**: dictionary of relevant experimental parameters including
            
            - *'hv'*: float, photon energy (eV), 
            
            - *'mfp'*: float, mean-free path (Angstrom),
            
            - *'resolution'*: dictionary for energy and momentum resolution:
                
                - *'dE'*: float, energy resolution (FWHM eV), 
                
                - *'dk'*: float, momentum resolution (FWHM 1/Angstrom)
                            
            - *'T'*: float, Temperature of sample (Kelvin)
            
            
            - *'cube'*: dictionary momentum and energy domain
            (*'kz'* as float, all others ( *'X'* , *'Y'* , *'E'* ) are list
            or tuple of floats Xo,Xf,dX)
                
    *optional args*:
        
        In addition to the keys above, *ARPES_dict* can also be fed the following:
            
            - *'spin'*: spin-ARPES measurement, list [+/-1,np.array([a,b,c])] 
            with the numpy array indicating the spin-projection 
            direction (with respect to) experimental frame.
            
            - *'rad_type'*: string, radial wavefunctions, c.f. *chinook.rad_int.py* for details
            
            - *'threads'*: int, number of threads on which to calculate the matrix elements. 
            Requires very large calculation to see improvement over single core.
            
            - *'slab'*: boolean, will truncate the eigenfunctions beyond the penetration depth (specifically 4x penetration depth), default is False

            - *'ang'*: float, rotation of sample about normal emission i.e. z-axis (radian), default is 0.0

            - *'W'*: float, work function (eV), default is 4.0 

    
    ***            
    '''
    def __init__(self,TB,ARPES_dict):
        self.TB = TB
        
        if sum([o.spin for o in self.TB.basis])<len(self.TB.basis):
            self.spin = True
        else:
            self.spin = False
        try:
            self.cube = (ARPES_dict['cube']['X'],ARPES_dict['cube']['Y'],ARPES_dict['cube']['E'])
            self.coord_type = 'momentum'
        except KeyError:
            try:
                self.cube = (ARPES_dict['cube']['Tx'],ARPES_dict['cube']['Ty'],ARPES_dict['cube']['E'])
                self.coord_type = 'angle'
            except KeyError:
                print('Error: must pass either a momentum (X,Y,E) or angle (Tx,Ty,E) range of interest to "cube" key of input dictionary.')
                return None

        self.hv = ARPES_dict['hv']
        self.dE = ARPES_dict['resolution']['E']/np.sqrt(8*np.log(2)) #energy resolution FWHM
        self.dk = ARPES_dict['resolution']['k']/np.sqrt(8*np.log(2)) #momentum resolution FWHM
        self.maps = []
        self.SE_args = ARPES_dict['SE']
        try:
            self.mfp = ARPES_dict['mfp'] #photoelectron mean free path for escape
        except KeyError:
            self.mfp = 10.0
        try:
            self.kz = ARPES_dict['cube']['kz']    
        except KeyError:
            self.kz = 0.0    
        try:
            self.W = ARPES_dict['W']
        except KeyError:
            self.W = 4.0
        try:
            self.Vo = ARPES_dict['Vo']
        except KeyError:
            self.Vo = -1
        try:
            self.ang = ARPES_dict['angle']
        except KeyError:
            self.ang = 0.0
        try:
            self.pol = ARPES_dict['pol']
        except KeyError:
            self.pol = np.array([1,0,0])
        try:
            self.T = ARPES_dict['T']
        except KeyError:
            self.T = -1
        try:
            self.sarpes = ARPES_dict['spin']
        except KeyError:
            self.sarpes = None
        try:
            self.rad_type = ARPES_dict['rad_type']
        except KeyError:
            self.rad_type = 'slater'
        try:
            self.rad_args = ARPES_dict['rad_args']
        except KeyError:
            self.rad_args = None
            
        try:
            self.phase_shifts= ARPES_dict['phase_shifts']
        except KeyError:
            self.phase_shifts = None
        try:
            self.slit = ARPES_dict['slit']
        except KeyError:
            self.slit = 'H'
        try:
            self.truncate = ARPES_dict['slab']
        except KeyError:
            self.truncate = False
        try:
            self.threads = ARPES_dict['threads']
        except KeyError:
            self.threads = 0
            
    def update_pars(self,ARPES_dict,datacube=False):
        '''
        Several experimental parameters can be updated without re-calculating 
        the ARPES intensity explicitly. Specifically here, we can update 
        resolution in both energy and momentum, as well as temperature,
        spin-projection, self-energy function, and polarization.
        
        *args*:
            - **ARPES_dict**: dictionary, specifically containing
                
                - *'resolution'*: dictionary with 'E':float and 'k':float

                - *'T'*: float, temperature, a negative value will suppress the Fermi function

                - *'spin'*: list of [int, numpy array of 3 float] indicating projection and spin vector

                - *'SE'*: various types accepted, see *SE_gen* for details

                - *'pol'*: numpy array of 3 complex float, polarization of light
        
        *kwargs*:
            - **datacube**: bool, if updating in *spectral*, only the above  can be changed. If instead, updating
            at the start of *datacube*, can also pass:
                - **hv**: float, photon energy, eV
                
                - **ang**: float, sample orientation around normal, radiants
                
                - **rad_type**: string, radial integral type
                
                - **rad_args**: various datatype, see *radint_lib* for details
                
                - **kz**: float, out-of-plane momentum, inverse Angstrom
                
                - **mfp**: float, mean-free path, Angstrom
        
        '''
        if 'resolution' in ARPES_dict.keys():
            try:
                self.dE = ARPES_dict['resolution']['E']/np.sqrt(8*np.log(2)) #energy resolution FWHM
                self.dk = ARPES_dict['resolution']['k']/np.sqrt(8*np.log(2)) #momentum resolution FWHM
            except KeyError:
                print('Energy "E" and momentum "k" resolutions not passed in "resolution" dictionary. \n Retaining original values.')
        if 'T' in ARPES_dict.keys():
            self.T = ARPES_dict['T']
        if 'spin' in ARPES_dict.keys():
            self.sarpes = ARPES_dict['spin']
        if 'SE' in ARPES_dict.keys():
            self.SE_args = ARPES_dict['SE']
        if 'pol' in ARPES_dict.keys():
            self.pol = ARPES_dict['pol']
        if 'slit' in ARPES_dict.keys():
            self.slit = ARPES_dict['slit']
        if datacube:
            if 'hv' in ARPES_dict.keys():
                self.hv = ARPES_dict['hv']
            if 'rad_type' in ARPES_dict.keys():
                self.rad_type = ARPES_dict['rad_type']
            if 'rad_args' in ARPES_dict.keys():
                self.rad_args = ARPES_dict['rad_args']
            if 'Vo' in ARPES_dict.keys():
                self.Vo = ARPES_dict['Vo']
            if 'kz' in ARPES_dict.keys():
                self.kz = ARPES_dict['kz']
            if 'mfp' in ARPES_dict.keys():
                self.mfp = ARPES_dict['mfp']

    
    
    def diagonalize(self):
        '''
        Diagonalize the Hamiltonian over the desired range of momentum, reshaping the 
        band-energies into a 1-dimensional array. If the user has not selected a energy
        grain for calculation, automatically calculate this.

        *return*:
            None, however *experiment* attributes *X*, *Y*, *ph*, *TB.Kobj*, *Eb*, *Ev*, *cube*
            are modified.
        '''
        if self.Vo>0:
            
            kn = (self.hv-self.W)
            Vo_args =[self.Vo,kn]
        else:
            Vo_args = None
            
        if self.coord_type=='momentum':
            x = np.linspace(*self.cube[0])
            y = np.linspace(*self.cube[1])
            X,Y = np.meshgrid(x,y)
        
            self.X = X
            self.Y = Y
            
            k_arr,self.ph = K_lib.kmesh(self.ang,self.X,self.Y,self.kz,Vo_args)      
            
    
        elif self.coord_type=='angle':
            k_arr = tilt.gen_kpoints(self.hv-self.W,(self.cube[0][2],self.cube[1][2]),self.cube[0][:2],self.cube[1][:2],self.kz)
    
            self.X = np.reshape(k_arr[:,0],(self.cube[1][2],self.cube[0][2]))
            self.Y = np.reshape(k_arr[:,1],(self.cube[1][2],self.cube[0][2]))
    
            self.ph = np.arctan2(k_arr[:,1],k_arr[:,0])
    
        self.TB.Kobj = K_lib.kpath(k_arr)
        self.Eb,self.Ev = self.TB.solve_H()
        
        if len(self.cube[2])==2:
            #user only passed the energy limits, not the grain--automate generation of the grain size
            band_dE_max = find_mean_dE(self.Eb)
            NE_pts = int(10*(self.cube[2][1]-self.cube[2][0])/band_dE_max)
            self.cube[2].append(NE_pts)
        
        self.Eb = np.reshape(self.Eb,(np.shape(self.Eb)[-1]*np.shape(self.X)[0]*np.shape(self.X)[1])) 
        

        
    def truncate_model(self):
        '''
        For slab calculations, the number of basis states becomes a significant memory load,
        as well as a time bottleneck. In reality, an ARPES calculation only needs the small
        number of basis states near the surface. Then for slab-calculations, we can truncate
        the basis and eigenvectors used in the calculation to dramatically improve our
        capacity to perform such calculations. We keep all eigenvectors, but retain only the
        projection of the basis states within 2*the mean free path of the surface. The 
        states associated with this projection are retained, while remainders are not.
        
        *return*:
            - **tmp_basis**: list, truncated subset of the basis' orbital objects
            
            - **Evec**: numpy array of complex float corresponding to the truncated eigenvector
             array containing only the surface-projected wavefunctions        
        '''
        depths = np.array([abs(oi.depth) for oi in self.basis])
        i_start = np.where(depths<4*self.mfp)[0][0]

        tmp_basis = []
        #CASE 1: BASIS INCLUDES BOTH SPIN DOF
        if self.spin:
            
            switch = (int(len(self.basis)/2))
            tmp_basis = self.basis[i_start:switch] + self.basis[(switch+i_start):]
          
            
            Evec = np.zeros((np.shape(self.Ev)[0],len(tmp_basis),np.shape(self.Ev)[-1]),dtype=complex)
            
            Evec[:,:(switch-i_start),:] =self.Ev[:,i_start:switch,:]
            Evec[:,(switch-i_start):,:] = self.Ev[:,(switch+i_start):,:]
        #CASE 2: BASIS IS SPINLESS
        else:
            
            tmp_basis = self.basis[i_start:]
            Evec=self.Ev[:,i_start:,:]
        return tmp_basis,Evec
    
    def rot_basis(self):
        '''
        Rotate the basis orbitals and their positions in the lab frame to be consistent with the
        experimental geometry
        
        *return*:
            - list of orbital objects, representing a rotated version of the original basis if the 
            angle is finite. Otherwise, just return the original basis.
        '''
        tmp_base = []
        if abs(self.ang)>0.0:
            for o in range(len(self.TB.basis)):
                oproj = np.copy(self.TB.basis[o].proj)
                l = self.TB.basis[o].l
                nproj,_ = olib.rot_projection(l,oproj,[np.array([0,0,1]),self.ang])
                tmp = self.TB.basis[o].copy()
                tmp.proj = nproj
                tmp_base.append(tmp)

            return tmp_base
        else:
            return self.TB.basis
    
###############################################################################    
###############################################################################    
##################  MAIN MATRIX ELEMENT EVALUATION  ###########################
###############################################################################
############################################################################### 
    
    def datacube(self,ARPES_dict=None):
        '''
        This function computes the photoemission matrix elements.
        Given a kmesh to calculate the photoemission over, the mesh is reshaped to an nx3 array and the Hamiltonian
        diagonalized over this set of k points. The matrix elements are then calculated for each 
        of these E-k points
    
        *kwargs*:
            - **ARPES_dict**: can optionally pass a dictionary of experimental parameters, to update those defined
            in the initialization of the *experiment* object.

        *return*:
            - boolean, True if function finishes successfully.
        '''      
        if ARPES_dict is not None:
            self.update_pars(ARPES_dict,True)

        self.basis = self.rot_basis()

        print('Initiate diagonalization: ')
        self.diagonalize()
        print('Diagonalization Complete.')
        nstates = len(self.basis)
        if self.truncate:

            self.basis,self.Ev = self.truncate_model()
            

        dE = (self.cube[2][1]-self.cube[2][0])/self.cube[2][2]            
        dig_range = (self.cube[2][0]-5*dE,self.cube[2][1]+5*dE)
           

        self.pks = np.array([[i,np.floor(np.floor(i/nstates)/np.shape(self.X)[1]),np.floor(i/nstates)%np.shape(self.X)[1],self.Eb[i]] for i in range(len(self.Eb)) if dig_range[0]<=self.Eb[i]<=dig_range[-1]])
        if len(self.pks)==0:
            raise ValueError('ARPES Calculation Error: no states found in energy window. Consider refining the region of interest')

        self.Mk = np.zeros((len(self.pks),2,3),dtype=complex)

        kn = (2.*me/hb**2*(self.hv+self.pks[:,3]-self.W)*q)**0.5*A

        self.th = np.array([np.arccos((kn[i]**2-self.X[int(self.pks[i,1]),int(self.pks[i,2])]**2-self.Y[int(self.pks[i,1]),int(self.pks[i,2])]**2)**0.5/kn[i]) if (kn[i]**2-self.X[int(self.pks[i,1]),int(self.pks[i,2])]**2-self.Y[int(self.pks[i,1]),int(self.pks[i,2])]**2)>=0 else -1 for i in range(len(self.pks))])

        
        self.prefactors = np.array([o.sigma*np.exp((-0.5/abs(self.mfp))*abs(o.depth)) for o in self.basis])
        self.Largs,self.Margs,Gmats,self.orbital_pointers = all_Y(self.basis) 
        self.Gbasis = Gmats[self.orbital_pointers]
        self.proj_arr = projection_map(self.basis)
        
        rad_dict = {'hv':self.hv,'W':self.W,'rad_type':self.rad_type,'rad_args':self.rad_args,'phase_shifts':self.phase_shifts}
        self.Bfuncs,self.radint_pointers = radint_lib.make_radint_pointer(rad_dict,self.basis,dig_range)


        print('Begin computing matrix elements: ')
        
        valid_indices = np.array([i for i in range(len(self.pks)) if (self.th[i]>=0)])# and self.cube[2][0]<=self.pks[i][3]<=self.cube[2][1])])

        if self.threads>0:
            self.thread_Mk(self.threads,valid_indices)
        else:
            self.serial_Mk(valid_indices)

        print('\nDone matrix elements')


        return True
    
    
    
    
    
    def M_compute(self,i):
        '''
        The core method called during matrix element computation.
        
        *args*:
            - **i**: integer, index and energy of state
        
        *return*:
            - **Mtmp**: numpy array (2x3) of complex float corresponding to the matrix element
              projection for dm = -1,0,1 (columns) and spin down or up (rows) for a given
              state in k and energy.
        '''
        nstates = len(self.TB.basis)
        phi = self.ph[int(self.pks[i,0]/nstates)]

        th = self.th[i]
        Ylm_calls = Yvect(self.Largs,self.Margs,th,phi)[self.orbital_pointers]
        
        Mtmp = np.zeros((2,3),dtype=complex)
        B_eval = np.array([[b[0](self.pks[i,3]),b[1](self.pks[i,3])] for b in self.Bfuncs])
        pref = np.einsum('i,ij->ij',np.einsum('i,i->i',self.prefactors,self.Ev[int(self.pks[i,0]/nstates),:,int(self.pks[i,0]%nstates)]),B_eval[self.radint_pointers])  
        Gtmp = np.einsum('ij,ijkl->ikl',self.proj_arr,np.einsum('ijkl,ijkl->ijkl',Ylm_calls,self.Gbasis))
        if self.spin:
            Mtmp[0,:] = np.einsum('ij,ijk->k',pref[:int(len(self.basis)/2)],Gtmp[:int(len(self.basis)/2)])
            Mtmp[1,:] = np.einsum('ij,ijk->k',pref[int(len(self.basis)/2):],Gtmp[int(len(self.basis)/2):])
        else:
            Mtmp[0,:] = np.einsum('ij,ijk->k',pref,Gtmp)
                   
        return Mtmp
    
    
    
    
    def serial_Mk(self,indices):
        '''
        Run matrix element on a single thread, directly modifies the *Mk* attribute.
        
        *args*:
            - **indices**: list of all state indices for execution; restricting states
             in *cube_indx* to those within the desired window     
        '''
        for ii in indices:
            sys.stdout.write('\r'+progress_bar(ii+1,len(self.pks)))

            self.Mk[ii,:,:]+=self.M_compute(ii)
        
    def thread_Mk(self,N,indices):
        '''
        Run matrix element on *N* threads using multiprocess functions, directly modifies the *Mk*
        attribute.
        
        NOTE 21/2/2019 -- this has not been optimized to show any measureable improvement over serial execution.
        May require a more clever way to do this to get a proper speedup.
        
        *args*:
            - **N**: int, number of threads
            
            - **indices**: list of int, all state indices for execution; restricting 
            states in cube_indx to those within the desired window.
        '''
        div = int(len(indices)/N)
        pool = ThreadPool(N)
        results = np.array(pool.map(self.Mk_wrapper,[indices[ii*div:(ii+1)*div] for ii in range(N)]))
        pool.close()
        pool.join()
        results = results.reshape(len(indices),2,3)
        self.Mk[indices] = results
        
        
    def Mk_wrapper(self,ilist):
        '''
        Wrapper function for use in multiprocessing, to run each of the processes
        as a serial matrix element calculation over a sublist of state indices.

        *args*:
            - **ilist**: list of int, all state indices for execution.

        *return*:
            - **Mk_out**: numpy array of complex float with shape (len(ilist), 2,3)
        '''
        Mk_out = np.zeros((len(ilist),2,3),dtype=complex)
        for ii in list(enumerate(ilist)):
            Mk_out[ii[0],:,:] += self.M_compute(ii[1])
        return Mk_out
    
    


###############################################################################    
###############################################################################    
####################### DATA VIEWING  #########################################
###############################################################################
############################################################################### 
    def SE_gen(self):
        '''
        Self energy arguments are passed as a list, which supports mixed-datatype.
        The first entry in list is a string, indicating the type of self-energy, 
        and the remaining entries are the self-energy. 
        
        *args*:
            - **SE_args**: list, first entry can be 'func', 'poly', 'constant', or 'grid'
            indicating an executable function, polynomial factors, constant, or a grid of values
        
        *return*:
            - SE, numpy array of complex float, with either shape of the datacube,
            or as a one dimensional array over energy only.
        '''
        
        w = np.linspace(*self.cube[2])
        
        if self.SE_args[0] == 'func':
            kx = np.linspace(*self.cube[0])
            ky = np.linspace(*self.cube[1])
            X,Y,W = np.meshgrid(kx,ky,w)
            try:
                SE = self.SE_args[1](X,Y,W)
            except TypeError:
                print('Using local (k-independent) self-energy.')
                SE = self.SE_args[1](w)
        elif self.SE_args[0] == 'grid':
            SE = np.interp(w,self.SE_args[1],self.SE_args[2])
        elif self.SE_args[0] == 'poly':
            SE = -1.0j*abs(poly(w,self.SE_args[1:]))
        elif self.SE_args[0] == 'constant':
            SE = -1.0j*abs(self.SE_args[1])

        return SE
            
    def smat_gen(self,svector=None):
        '''
        Define the spin-projection matrix related to a spin-resolved ARPES experiment.
        
        *return*:
            - **Smat**: numpy array of 2x2 complex float corresponding to Pauli operator along the desired direction
        '''
        try:
            sv = svector/np.linalg.norm(svector)
        except TypeError:
            try:              
                sv = self.sarpes[1]/np.linalg.norm(self.sarpes[1])
            except IndexError:
                print('ERROR: Invalid spin-entry. See documentation for ARPES_lib.experiment')
                return None
        th = np.arccos(sv[2])
        ph = np.arctan2(sv[1],sv[0])
        if abs(self.ang)>0:
            ph+=self.ang
        Smat = np.array([[np.cos(th/2),np.exp(-1.0j*ph)*np.sin(th/2)],[np.sin(th/2),-np.exp(-1.0j*ph)*np.cos(th/2)]])
        return Smat
        
    def sarpes_projector(self):
        '''
        For use in spin-resolved ARPES experiments, project the computed
        matrix element values onto the desired spin-projection direction.
        In the event that the spin projection direction is not along the 
        standard out-of-plane quantization axis, we rotate the matrix elements
        computed into the desired basis direction.
            
        *return*:
            - **spin_projected_Mk**: numpy array of complex float with same
            shape as *Mk*
        '''
        if self.coord_type == 'momentum':
            Smat = self.smat_gen()
            spin_projected_Mk = np.einsum('ij,kjl->kil',Smat,self.Mk)
            
        elif self.coord_type == 'angle':
            if self.slit=='H':
                th =0.5*(self.cube[0][0]+self.cube[0][1])
                phvals = np.linspace(*self.cube[1])
                pk_index = 1
                
                Rmats = np.array([np.matmul(rotlib.Rodrigues_Rmat(np.array([1,0,0]),-ph),rotlib.Rodrigues_Rmat(np.array([0,1,0]),-th)) for ph in phvals])
            elif self.slit=='V':
                ph = 0.5*(self.cube[1][0]+self.cube[1][1])
                thvals = np.linspace(*self.cube[0])
                Rmats = np.array([np.matmul(rotlib.Rodrigues_Rmat(np.array([0,np.cos(-ph),np.sin(-ph)]),-th),rotlib.Rodrigues_Rmat(np.array([1,0,0])-ph)) for th in thvals])
                pk_index = 2
                
            svectors = np.einsum('ijk,k->ij',Rmats,self.sarpes[1])
            Smats = np.array([self.smat_gen(sv) for sv in svectors])
            all_mats = Smats[np.array([int(self.pks[i,pk_index]) for i in range(len(self.pks))])]
            spin_projected_Mk = np.einsum('ijk,ikl->ijl',all_mats,self.Mk)
            
        return spin_projected_Mk
    
    def gen_all_pol(self):
        '''
        Rotate polarization vector, as it appears for each angle in the experiment.
        Assume that it only rotates with THETA_y (vertical cryostat), and that the polarization
        vector defined by the user relates to centre of THETA_x axis. 
        Right now only handles zero vertical rotation (just tilt)
        
        *return*:
            - numpy array of len(expmt.cube[1]) x 3 complex float, rotated polarization vectors 
            expressed in basis of spherical harmonics
        '''
        
        if self.slit=='H':
            th =0.5*(self.cube[0][0]+self.cube[0][1])
            phvals = np.linspace(*self.cube[1])
            Rmats = np.array([np.matmul(rotlib.Rodrigues_Rmat(np.array([1,0,0]),-ph),rotlib.Rodrigues_Rmat(np.array([0,1,0]),-th)) for ph in phvals])
            pk_index = 1
            
        elif self.slit=='V':
            ph = 0.5*(self.cube[1][0]+self.cube[1][1])
            thvals = np.linspace(*self.cube[0])
            Rmats = np.array([np.matmul(rotlib.Rodrigues_Rmat(np.array([0,np.cos(-ph),np.sin(-ph)]),-th),rotlib.Rodrigues_Rmat(np.array([1,0,0]),-ph)) for th in thvals])
            pk_index = 2
            
        rot_pols = np.einsum('ijk,k->ij',Rmats,self.pol)
        rot_pols_sph = pol_2_sph(rot_pols)
        peak_pols = np.array([rot_pols_sph[int(self.pks[i,pk_index])] for i in range(len(self.pks))])
        return peak_pols

    def T_distribution(self):
        '''
        Compute the Fermi-distribution for a fixed temperature, over the domain of energy of interest

        *return*:
            - **fermi**: numpy array of float, same length as energy domain array defined by *cube[2]* attribute.
        '''
        if np.sign(self.T)>-1:
            fermi = vf(np.linspace(*self.cube[2])/(kb*self.T/q))
        else:
            fermi = np.ones(self.cube[2][2])
        return fermi

    def spectral(self,ARPES_dict=None,slice_select=None,add_map = False,plot_bands=False,ax=None):
        
        '''
        Take the matrix elements and build a simulated ARPES spectrum. 
        The user has several options here for the self-energy to be used,  c.f. *SE_gen()* for details.
        Gaussian resolution broadening is the last operation performed, to be consistent with the
        practical experiment. *slice_select* instructs the method to also produce a plot of the designated
        slice through momentum or energy. If this is done, the function also returns the associated matplotlib.Axes
        object for further manipulation of the plot window.
        
        *kwargs*:
            - **ARPES_dict**: dictionary, experimental configuration. See *experiment.__init__* and *experiment.update_pars()*            
            
            - **slice_select**: tuple, of either (int,int) or (str,float) format. If (int,int), first is axis index (0,1,2 for x,y,E) and the second is the index of the array. More useful typically is (str,float) format, with str as 'x', 'kx', 'y', 'ky', 'E', 'w' and the float the value requested. It will find the index along this direction closest to the request. Note the strings are not case-sensitive.
            
            - **add_map**: boolean, add intensity map to list of intensity maps. If true, a list of intensity objects is appended, otherwise, the intensity map is overwritten
            
            - **plot_bands**: boolean, plot bandstructure from tight-binding over the intensity map
            
            - **ax**: matplotlib Axes, only relevant if **slice_select**, option to pass existing Axes to plot onto
        
        *return*:
            - **I**: numpy array of float, raw intensity map.

            - **Ig**: numpy array of float, resolution-broadened intensity map.
            
            - **ax**: matplotlib Axes, for further modifications to plot only if **slice_select** True
        '''
        if not hasattr(self,'Mk'):
            self.datacube()
            
        if ARPES_dict is not None:

            self.update_pars(ARPES_dict)
        
        
        if self.sarpes is not None:
            spin_Mk = self.sarpes_projector()
            if self.coord_type == 'momentum':
                pol = pol_2_sph(self.pol)

                M_factor = np.power(abs(np.einsum('ij,j->i',spin_Mk[:,int((self.sarpes[0]+1)/2),:],pol)),2)
            elif self.coord_type == 'angle':
                all_pol = self.gen_all_pol()
                M_factor = np.power(abs(np.einsum('ij,ij->i',spin_Mk[:,int((self.sarpes[0]+1)/2),:],all_pol)),2)
        else:
            if self.coord_type == 'momentum':
                pol = pol_2_sph(self.pol)

                M_factor = np.sum(np.power(abs(np.einsum('ijk,k->ij',self.Mk,pol)),2),axis=1)
            elif self.coord_type == 'angle':
                all_pol = self.gen_all_pol()
                M_factor = np.sum(np.power(abs(np.einsum('ijk,ik->ij',self.Mk,all_pol)),2),axis=1)                
        
        SE = self.SE_gen()
        fermi = self.T_distribution()    
        w = np.linspace(*self.cube[2])
        
        I = np.zeros((self.cube[1][2],self.cube[0][2],self.cube[2][2]))

        if np.shape(SE)==np.shape(I):
            SE_k = True
        else:
            SE_k = False

        for p in range(len(self.pks)):

            if not SE_k:
                I[int(np.real(self.pks[p,1])),int(np.real(self.pks[p,2])),:] += M_factor[p]*np.imag(-1./(np.pi*(w-self.pks[p,3]-(SE-0.0005j))))*fermi
            else:
                I[int(np.real(self.pks[p,1])),int(np.real(self.pks[p,2])),:]+= M_factor[p]*np.imag(-1./(np.pi*(w-self.pks[p,3]-(SE[int(np.real(self.pks[p,1])),int(np.real(self.pks[p,2])),:]-0.0005j))))*fermi 


        kxg = (self.cube[0][2]*self.dk/(self.cube[0][1]-self.cube[0][0]) if abs(self.cube[0][1]-self.cube[0][0])>0 else 0)
        kyg = (self.cube[1][2]*self.dk/(self.cube[1][1]-self.cube[1][0]) if abs(self.cube[1][1]-self.cube[1][0])>0 else 0)
        wg = (self.cube[2][2]*self.dE/(self.cube[2][1]-self.cube[2][0]) if abs(self.cube[2][1]-self.cube[2][0])>0 else 0)

        Ig = nd.gaussian_filter(I,(kyg,kxg,wg))
        
        if slice_select!=None:
            ax_img = self.plot_intensity_map(Ig,slice_select,plot_bands,ax)
        
        if add_map:
            self.maps.append(imap.intensity_map(len(self.maps),Ig,self.cube,self.kz,self.T,self.hv,self.pol,self.dE,self.dk,self.SE_args,self.sarpes,self.ang))
        else:
            self.maps = [imap.intensity_map(len(self.maps),Ig,self.cube,self.kz,self.T,self.hv,self.pol,self.dE,self.dk,self.SE_args,self.sarpes,self.ang)]
        if slice_select:
            return I,Ig,ax_img
        else:
            return I,Ig
    
    def gen_imap(self,I_arr):
        new_map = imap.intensity_map(len(self.maps),I_arr,self.cube,self.kz,self.T,self.hv,self.pol,self.dE,self.dk,self.SE_args,self.sarpes,self.ang)
        return new_map


    def plot_intensity_map(self,plot_map,slice_select,plot_bands=False,ax_img=None):
         '''
        Plot a slice of the intensity map computed in *spectral*. The user selects either
        an array index along one of the axes, or the fixed value of interest, allowing
        either integer, or float selection.
        
        *args*:
            - **plot_map**: numpy array of shape (self.cube[0],self.cube[1],self.cube[2]) of float

            - **slice_select**: list of either [int,int] or [str,float], corresponding to 
            dimension, index or label, value. The former option takes dimensions 0,1,2 while
            the latter can handle 'x', 'kx', 'y', 'ky', 'energy', 'w', or 'e', and is not
            case-sensitive.
            
            - **plot_bands**: boolean, option to overlay a constant-momentum cut with
            the dispersion calculated from tight-binding
            
            - **ax_img**: matplotlib Axes, for option to plot onto existing Axes

        *return*:

            - **ax_img**: matplotlib axis object
         '''
         if ax_img is None:
             fig,ax_img = plt.subplots()
             fig.set_tight_layout(False)
         

         if type(slice_select[0]) is str:
             str_opts = [['x','kx'],['y','ky'],['energy','w','e']]
             dim = 0
             for i in range(3):
                 if slice_select[0].lower() in str_opts[i]:
                     dim = i
             x = np.linspace(*self.cube[dim])
             index = np.where(abs(x-slice_select[1])==abs(x-slice_select[1]).min())[0][0]
             slice_select = [dim,int(index)]
             
        
       
        
        #new option
         index_dict = {2:(0,1),1:(2,0),0:(2,1)}
         
         X,Y = np.meshgrid(np.linspace(*self.cube[index_dict[slice_select[0]][0]]),np.linspace(*self.cube[index_dict[slice_select[0]][1]]))
         limits = np.zeros((3,2),dtype=int)
         limits[:,1] = np.shape(plot_map)[1],np.shape(plot_map)[0],np.shape(plot_map)[2]
         limits[slice_select[0]] = [slice_select[1],slice_select[1]+1]

         
         ax_xlimit = (self.cube[index_dict[slice_select[0]][0]][0],self.cube[index_dict[slice_select[0]][0]][1])
         ax_ylimit = (self.cube[index_dict[slice_select[0]][1]][0],self.cube[index_dict[slice_select[0]][1]][1])
         plottable  = np.squeeze(plot_map[limits[1,0]:limits[1,1],limits[0,0]:limits[0,1],limits[2,0]:limits[2,1]])
         p = ax_img.pcolormesh(X,Y,plottable,cmap=cm.magma)
         if plot_bands and slice_select[0]!=2:
             k = np.linspace(*self.cube[index_dict[slice_select[0]][1]])
             if slice_select[0]==1:                 
                 indices = np.array([len(k)*slice_select[1] + ii for ii in range(len(k))])
             elif slice_select[0]==0:
                 indices = np.array([slice_select[1] + ii*self.cube[0][2] for ii in range(len(k))])
             for ii in range(len(self.TB.basis)):  
                 ax_img.plot(self.TB.Eband[indices,ii],k,alpha=0.4,c='w')
#            
         ax_img.set_xlim(*ax_xlimit)
         ax_img.set_ylim(*ax_ylimit)
         

         plt.colorbar(p,ax=ax_img)
         plt.tight_layout()

         return ax_img
        
    
    def plot_gui(self):
        '''
        Generate the Tkinter gui for exploring the experimental parameter-space
        associated with the present experiment.

        *args*:
            - **ARPES_dict**: dictionary of experimental parameters, c.f. the 
            *__init__* function for details.

        *return*:
            - **Tk_win**: Tkinter window.
        '''
        if tk_found:
            TK_win = Tk_plot.plot_intensity_interface(self)
        
            return TK_win
        else:
            print('This tool is not active without tkinter')
            return None
        
        
###############################################################################    
###############################################################################    
################### WRITE ARPES MAP TO FILE ###################################
###############################################################################
###############################################################################        
     
    def write_map(self,_map,directory):
        '''
        Write the intensity maps to a series of text files in the indicated directory.

        *args*:
            - **_map**: numpy array of float to write to file

            - **directory**: string, name of directory + the file-lead name 

        *return*:
            - boolean, True
        '''
        for i in range(np.shape(_map)[2]):   
            filename = directory + '_{:d}.txt'.format(i)
            self.write_Ik(filename,_map[:,:,i])
        return True

    def write_params(self,Adict,parfile):
        '''
        Generate metadata text file  associated with the saved map.
        
        *args*:
            - **Adict**: dictionary, ARPES_dict same as in above functions, containing
            relevant experimental parameters for use in saving the metadata associated
            with the related calculation.
            
            - **parfile**: string, destination for the metadata
        '''
        
        
        RE_pol = list(np.real(Adict['pol']))
        IM_pol = list(np.imag(Adict['pol']))
        with open(parfile,"w") as params:
            params.write("Photon Energy: {:0.2f} eV \n".format(Adict['hv']))
            params.write("Temperature: {:0.2f} K \n".format(Adict['T'][1]))
            params.write("Polarization: {:0.3f}+{:0.3f}j {:0.3f}+{:0.3f}j {:0.3f}+{:0.3f}j\n".format(RE_pol[0],IM_pol[0],RE_pol[1],IM_pol[1],RE_pol[2],IM_pol[2]))

            params.write("Energy Range: {:0.6f} {:0.6f} {:0.6f}\n".format(Adict['cube']['E'][0],Adict['cube']['E'][1],Adict['cube']['E'][2]))
            params.write("Kx Range: {:0.6f} {:0.6f} {:0.6f}\n".format(Adict['cube']['X'][0],Adict['cube']['X'][1],Adict['cube']['X'][2]))
            params.write("Ky Range: {:0.6f} {:0.6f} {:0.6f}\n".format(Adict['cube']['Y'][0],Adict['cube']['Y'][1],Adict['cube']['Y'][2]))
            params.write("Kz Value: {:0.6f}\n".format(Adict['cube']['kz']))
            try:
                params.write("Azimuthal Rotation: {:0.6f}\n".format(Adict['angle']))
            except ValueError:
                pass
            params.write("Energy Resolution: {:0.4f} eV\n".format(Adict['resolution']['E']))
            params.write("Momentum Resolution: {:0.4f} eV\n".format(Adict['resolution']['k']))
            params.write('Self Energy: '+'+'.join(['{:0.04f}w^{:d}'.format(Adict['SE'][i],i) for i in range(len(Adict['SE']))])+'\n')
            try:
                params.write("Spin Projection ({:s}): {:0.4f} {:0.4f} {:0.4f}\n".format(('up' if Adict['spin'][0]==1 else 'down'),Adict['spin'][1][0],Adict['spin'][1][1],Adict['spin'][1][2]))
            except TypeError:
                pass
            
        params.close()
    
    def write_Ik(self,filename,mat):
        '''
        Function for producing the textfiles associated with a 2 dimensional numpy array of float
        
        *args*:
            
            - **filename**: string indicating destination of file
            
            - **mat**: numpy array of float, two dimensional

        *return*:
            - boolean, True
        
        '''
        with open(filename,"w") as destination:
            for i in range(np.shape(mat)[0]):
                tmpline = " ".join(map(str,mat[i,:]))
                tmpline+="\n"
                destination.write(tmpline)
        destination.close()
        return True
    
    
###############################################################################    
###############################################################################    
######################## SUPPORT FUNCTIONS#####################################
###############################################################################
###############################################################################
def find_mean_dE(Eb):
    '''
    Find the average spacing between adjacent points along the dispersion calculated.
    
    *args*:
        - **Eb**: numpy array of float, eigenvalues
        
    *return*:
        - **dE_mean**: float, average difference between consecutive eigenvalues.
    '''
    
    dE_mean = abs(np.subtract(Eb[1:,:],Eb[:-1,:])).mean()
    return dE_mean      
        
def con_ferm(ekbt):      
    '''
    Typical values in the relevant domain for execution of the Fermi distribution will
    result in an overflow associated with 64-bit float. To circumvent, set fermi-function
    to zero when the argument of the exponential in the denominator is too large.

    *args*:
        - **ekbt**: float, (E-u)/kbT in terms of eV

    *return*:
        - **fermi**: float, evaluation of Fermi function.
    '''
    fermi = 0.0
    if ekbt<709:
        fermi = 1.0/(np.exp(ekbt)+1)
    return fermi


vf = np.vectorize(con_ferm)



def pol_2_sph(pol):
    '''
    return polarization vector in spherical harmonics -- order being Y_11, Y_10, Y_1-1.
    If an array of polarization vectors is passed, use the einsum function to broadcast over
    all vectors.

    *args*:
        - **pol**: numpy array of 3 complex float, polarization vector in Cartesian coordinates (x,y,z)

    *return*:
        - numpy array of 3 complex float, transformed polarization vector.
    '''
    M = np.sqrt(0.5)*np.array([[-1,1.0j,0],[0,0,np.sqrt(2)],[1.,1.0j,0]])
    if len(np.shape(pol))>1:
        return np.einsum('ij,kj->ik',M,pol).T
    else:
        return np.dot(M,pol)




def poly(input_x,poly_args):
    '''
    Recursive polynomial function.
    
    *args*:
        - **input_x**: float, int or numpy array of numeric type, input value(s) at which to evaluate the polynomial 
        
        - **poly_args**: list of coefficients, in INCREASING polynomial order i.e. [a_0,a_1,a_2] for y = a_0 + a_1 * x + a_2 *x **2
    
    *return*:
        - recursive call to *poly*, if *poly_args* is reduced to a single value, return explicit evaluation of the function.
        Same datatype as input, with int changed to float if *poly_args* are float, polynomial evaluated over domain of *input_x*
    '''
    if len(poly_args)==0:
        return 0
    else:
        return input_x**(len(poly_args)-1)*poly_args[-1] + poly(input_x,poly_args[:-1])
    
  

#        
        
    
def progress_bar(N,Nmax):
    '''
    Utility function, generate string to print matrix element calculation progress.

    *args*:
        - **N**: int, number of iterations complete

        - **Nmax**: int, total number of iterations to complete

    *return*:
        - **st**: string, progress status
    '''
    frac = N/Nmax
    st = ''.join(['|' for i in range(int(frac*30))])
    st = '{:30s}'.format(st)+'{:3d}%'.format(int(frac*100))
    return st


###############################################################################    
###############################################################################    
######################## ANGULAR INTEGRALS ####################################
###############################################################################
###############################################################################
        
def G_dic():
    '''
    Initialize the gaunt coefficients associated with all possible transitions relevant

    *return*:
        - **Gdict**: dictionary with keys as a string representing (l,l',m,dm) "ll'mdm" and values complex float.
        All unacceptable transitions set to zero.
    '''
    llp = [[l,lp] for l in range(4) for lp in ([l-1,l+1] if (l-1)>=0 else [l+1])]    

    llpmu = [[l[0],l[1],m,u] for l in llp for m in np.arange(-l[0],l[0]+1,1) for u in [-1,0,1]]
    keyvals = [[str(l[0])+str(l[1])+str(l[2])+str(l[3]), Ylm.gaunt(l[0],l[2],l[1]-l[0],l[3])] for l in llpmu]
    G_dict = dict(keyvals)
    
    for gi in G_dict:
        if np.isnan(G_dict[gi]):
            G_dict[gi]=0.0       
    return G_dict


def all_Y(basis):
    '''
    Build L-M argument array input arguments for every combination of l,m in the basis. The idea is for a given k-point to have a single call
    to evaluate all spherical harmonics at once. The pointer array orb_point is a list of lists, where for each projection in the basis, the integer
    in the list indicates which row (first axis) of the Ylm array should be taken. This allows for very quick access to the l+/-1, m+/-1,0 Ylm evaluation
    required.
    
    *args*:
        - **basis**: list of orbital objects

    *return*:
        - **l_args**: numpy array of int, of shape len(*lm_inds*),3,2, with the latter two indicating the final state orbital angular momentum

        - **m_args**: numpy array of int, of shape len(*lm_inds*),3,2, with the latter two indicating the final state azimuthal angular momentum

        - **g_arr**: numpy array of float, shape len(*lm_inds*),3,2, providing the related Gaunt coefficients.

        - **orb_point**: numpy array of int, matching the related sub-array of *l_args*, *m_args*, *g_arr* related to each orbital in basis
    '''
    maxproj = max([len(o.proj) for o in basis])
    Gvals = G_dic()
    lm_inds = []
    l_args = []
    m_args =[]
    g_arr = []
    orb_point = []
    for o in basis:
        point = np.zeros(maxproj)
        for pi in range(len(o.proj)):
            p = o.proj[pi]
            lm = (p[2],p[3])
            if lm not in lm_inds:
                Yarr = ((np.ones((3,2))*np.array([lm[0]-1,lm[0]+1])).T,(np.ones((2,3))*np.array([lm[1]-1,lm[1]+0,lm[1]+1])))
                l_args.append(Yarr[0])
                m_args.append(Yarr[1])
                g_arr.append(Gmat_make(lm,Gvals))
                lm_inds.append(lm)
            point[pi] = lm_inds.index(lm)
        orb_point.append(point)
    return np.array(l_args),np.array(m_args),np.array(g_arr),np.array(orb_point).astype(int)
    

def projection_map(basis):

    '''
    In order to improve efficiency, an array of orbital projections is generated, carrying all and each
    orbital projection for the elements of the model basis. As these do not in general have the same length,
    the second dimension of this array corresponds to the largest of the sets of projections associated with
    a given orbital. This will in practice remain a modest number of order 1, since at worst we assume f-orbitals,
    in which case the projection can be no larger than 7 long. So output will be at worst len(basis)x7 complex float

    *args*:
        - **basis**: list of orbital objects

    *return*:
        - **projarr**: numpy array of complex float

    '''
    
    maxproj = max([len(o.proj) for o in basis])
    projarr = np.zeros((len(basis),maxproj),dtype=complex)
    for ii in range(len(basis)):
        for pj in range(len(basis[ii].proj)):
            proj = basis[ii].proj[pj]
            projarr[ii,pj] = proj[0]+1.0j*proj[1]
    return projarr




Yvect = np.vectorize(Ylm.Y,otypes=[complex])

def Gmat_make(lm,Gdictionary):
    '''
    Use the dictionary of relevant Gaunt coefficients to generate a small 2x3 array of  
    float which carries the relevant Gaunt coefficients for a given initial state.

    *args*:
        - **lm**: tuple of 2 int, initial state orbital angular momentum and azimuthal angular momentum

        - **Gdictionary**: pre-calculated dictionary of Gaunt coefficients, with key-values associated with "ll'mdm"

    *return*:
        - **mats**: numpy array of float 2x3
    '''

    l  = int(lm[0])
    m = int(lm[1])
    mats = np.zeros((2,3))
    for lp in (-1,1):
        for u in range(-1,2): 
            try:
                mats[int((lp+1)/2),u+1] = Gdictionary['{:d}{:d}{:d}{:d}'.format(l,l+lp,m,u)]
            except KeyError:
                continue
    return mats
    
    




def gen_SE_KK(w,SE_args):
    '''
    The total self-energy is computed using Kramers' Kronig relations:
        
        The user can pass the self-energy in the form of either a callable function, a list of polynomial coefficients, or as a numpy array with shape Nx2 (with the first
        column an array of frequency values, and the second the values of a function). For the latter option, the user is responsible for ensuring that the function goes 
        to zero at the tails of the domain. In the former two cases, the 'cut' parameter is used to impose an exponential cutoff near the edge of the domain to ensure this 
        is the case. In all cases the input imaginary self-energy must be single-signed to ensure it is purely even function. It is forced to be negative in all cases to give
        a positive spectral function.
        With the input defined, along with the energy range of interest to the calculation, a MUCH larger domain (100x in the maximal extent of the energy region of interest) is defined
        wf. This is the domain over which we evaluate the Hilbert transform, which itself is carried out using:
        the scipy.signal.hilbert() function. This function acting on an array f: H(f(x)) -> f(x) + i Hf(x). It relies on the FFT performed on the product of the sgn(w) and F(w) functions,
        and then IFFT back so that we can use this to extract the real part of the self energy, given only the input.
        args:
            w -- numpy array energy values for the spectral peaks used in the ARPES simulation
            SE_args -- dictionary containing the 'imfunc' key value pair (values being either callable, list of polynomial prefactors (increasing order) or numpy array of energy and Im(SE) values)
                    -- for the first two options, a 'cut' key value pair is also required to force the function to vanish at the boundary of the Hilbert transform integration window.
        return: self energy as a numpy array of complex float. The indexing matches that of w, the spectral features to be plotted in the matrix element simulation.
    '''
    
    
    if ('imfunc' not in SE_args):
        print('Self-Energy Error: Incorrect Dictionary key inputs. User requires "imfunc" for functional form for imaginary part')
        print('Returning a constant array of Im(SE) = -0.01, Re(SE) = 0.0')
        return -0.01j*np.ones(len(w))
    else:

        if type(SE_args['imfunc'])==np.ndarray and np.shape(SE_args['imfunc'])[1]==2:
            wf = SE_args['imfunc'][:,0]
            imSE = SE_args['imfunc'][:,1]
        else:
            wlim = abs(w).max()
            wf = np.arange(-100*wlim,100*wlim,(w[1]-w[0]))
            if callable(SE_args['imfunc']):
                imSE = SE_args['imfunc'](wf)
                if np.real(imSE).max()==0.0:
                    print('Input WARNING: The imaginary part of the self-energy should be passed as real-valued function (i.e. suppress the 1.0j factor). Converting imaginary part to real float and proceeding.')
                    imSE = np.imag(imSE)
            elif type(SE_args['imfunc'])==list or type(SE_args['imfunc'])==tuple or type(SE_args['imfunc'])==np.ndarray:
                if np.real(SE_args['imfunc']).max()==0.0:
                    print('Input WARNING: Pass arguments for imaginary part of self-energy as real values. Passing imaginary part as the functional arguments')
                    SE_args['imfunc'] = np.imag(SE_args['imfunc'])
                imSE = abs(poly(wf,SE_args['imfunc']))
            else:
                print('Invalid self-energy input format. Please see ARPES_lib.gen_SE for further details on input parameters')
                print('Returning a constant array of Im(SE) = 0.01, Re(SE) = 0.0')
                return -0.01j*np.ones(len(w))
                
                 ### IMPOSE THE CUTOFF!
            if abs(SE_args['cut'])>wf[-1]:
                SE_args['cut'] = 0.9*wf[-1]
                print('WARNING: INVALID CUTOFF (BEYOND HILBERT TRANSFORM DOMAIN). CUTTING TO: {:0.02f}'.format(SE_args['cut']))
            wcut = np.where(abs(wf-abs(SE_args['cut']))==abs(wf-abs(SE_args['cut'])).min())[0][0],np.where(abs(wf+abs(SE_args['cut']))==abs(wf+abs(SE_args['cut'])).min())[0][0]
            cut_width = wf[5]-wf[0]
            imSE[:wcut[1]] = np.exp(-abs(wf[:wcut[1]]-wf[wcut[1]])/cut_width)*imSE[wcut[1]]
            imSE[wcut[0]:] = np.exp(-abs(wf[wcut[0]:]-wf[wcut[0]])/cut_width)*imSE[wcut[0]]
                
            ##Check that IM(Self Energy) is positive/negative semi-definite. If postive, make negative
        sign_imSE = np.sign(imSE)
        sign_imSE = sign_imSE[abs(sign_imSE)>0]
        if sum(sign_imSE)<len(sign_imSE):
            print('WARNING: Invalid definition of imaginary part of self energy--values must all be single-signed. Returning constant -0.01j as Self-energy.')
            return -0.01j*np.ones(len(w))
        if sign_imSE[0]>0: #imaginary part of self energy should be <0
            imSE*=-1

        SEf = hilbert(imSE)
        reSE = -SEf.imag
        imSE = SEf.real
        roi = np.where(wf<w.min())[0][-1]-10,np.where(wf>w.max())[0][0]+10

        im_interp = interp1d(wf[roi[0]:roi[1]],imSE[roi[0]:roi[1]])
        re_interp = interp1d(wf[roi[0]:roi[1]],reSE[roi[0]:roi[1]])

        return re_interp(w) + im_interp(w)*1.0j
    
    
###

    
