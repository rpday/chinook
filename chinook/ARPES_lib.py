#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 21:15:20 2017

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

import sys

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
import chinook.Ylm as Ylm 


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
                
            - *'ang'*: float, rotation of sample about normal emission i.e. z-axis (radian),
            
            - *'slab'*: boolean, will truncate the eigenfunctions beyond the penetration depth (specifically 4x penetration depth)
            
            - *'T'*: float, Temperature of sample (Kelvin)
            
            - *'W'*: float, work function (eV)
            
            - *'cube'*: dictionary momentum and energy domain
            (*'kz'* as float, all others ( *'X'* , *'Y'* , *'E'* ) are list
            or tuple of floats Xo,Xf,dX)
                
    *optional args*:
        
        In addition to the keys above, *ARPES_dict* can also be fed the following:
            
            - *'spin'*: spin-ARPES measurement, list [+/-1,np.array([a,b,c])] 
            with the numpy array indicating the spin-projection 
            direction (with respect to) experimental frame.
            If not spin-ARPES, use None
            
            - *'rad_type'*: string, radial wavefunctions, c.f. *chinook.rad_int.py* for details
            
            - *'threads'*: int, number of threads on which to calculate the matrix elements. 
            Requires very large calculation to see improvement over single core.
            
            
            
    
    ***            
    '''
    def __init__(self,TB,ARPES_dict):
        self.TB = TB
        if sum([o.spin for o in self.TB.basis])<len(self.TB.basis):
            self.spin = True
        else:
            self.spin = False
        self.hv = ARPES_dict['hv']
        self.mfp = ARPES_dict['mfp'] #photoelectron mean free path for escape
        self.dE = ARPES_dict['resolution']['E']/np.sqrt(8*np.log(2)) #energy resolution FWHM
        self.dk = ARPES_dict['resolution']['k']/np.sqrt(8*np.log(2)) #momentum resolution FWHM
        self.ang = ARPES_dict['angle']
        self.T = ARPES_dict['T']
        self.W = ARPES_dict['W']
        self.cube = (ARPES_dict['cube']['X'],ARPES_dict['cube']['Y'],ARPES_dict['cube']['E'])
        self.kz = ARPES_dict['cube']['kz']
        self.SE_args = ARPES_dict['SE']
        if 'rad_type' in ARPES_dict.keys():
            self.rad_type = ARPES_dict['rad_type']
        else:
            self.rad_type = 'slater'
        try:
            self.truncate = ARPES_dict['slab']
        except KeyError:
            self.truncate = False
            
    def update_pars(self,ARPES_dict):
        '''
        Several experimental parameters can be updated without re-calculating 
        the ARPES intensity explicitly: only a change of photon energy and
        domain of interest require updating the matrix elements.
        
        
        '''
        if 'resolution' in ARPES_dict.keys():
            self.update_resolution(ARPES_dict['resolution'])
        if 'T' in ARPES_dict.keys():
            self.T = ARPES_dict['T']
        if 'spin' in ARPES_dict.keys():
            self.sarpes = ARPES_dict['spin']

    
    
    def diagonalize(self):
        '''
        Diagonalize the Hamiltonian over the desired range of momentum
        args:
            X,Y,kz -- momentum relevant to the calculation
        return:
            None, several attributes to the experiment object are modified/defined
        '''
        
        x = np.linspace(*self.cube[0])
        y = np.linspace(*self.cube[1])
        X,Y = np.meshgrid(x,y)
        
        self.X = X
        self.Y = Y
        
        k_arr,self.ph = K_lib.kmesh(self.ang,self.X,self.Y,self.kz)      
    
        self.TB.Kobj = K_lib.kpath(k_arr)
        self.Eb,self.Ev = self.TB.solve_H()
        
        if len(self.cube[2])==2:
            #user only passed the energy limits, not the grain--automate generation of the grain size
            band_dE_max = find_max_dE(self.Eb)
            NE_pts = int(10*(self.cube[2][1]-self.cube[2][0])/band_dE_max)
            self.cube[2].append(NE_pts)
        
        self.Eb = np.reshape(self.Eb,(np.shape(self.Eb)[-1]*np.shape(self.X)[0]*np.shape(self.X)[1])) 
        
        
        
    def truncate_model(self):
        '''
        For slab calculations, the number of basis states becomes a significant memory load, as well as a time bottleneck.
        In reality, an ARPES calculation only needs the small number of basis states near the surface. Then for slab-calculations,
        we can truncate the basis and eigenvectors used in the calculation to dramatically improve our capacity to perform such calculations
        We keep all eigenvectors, but retain only the projection of the basis states within 2*the mean free path of the surface. The states associated
        with this projection are retained, while remainders are not
        args:
            local_basis -- ARPES intensity is computed with a local copy of the TB basis to avoid scrambling data. This is passed here to be modified
        return:
            tmp_basis -- truncated list of orbital objects
            Evec -- numpy array of complex float --truncated eigenvector array containing only the surface-projected wavefunctions
        
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
    
    def update_resolution(self,resdict):
        '''
        Update the attributes relating to resolution when a parameter dictionary is passed with different values
        args:
            resdict --dictionary with 'E' and 'k' keys relating to the FWHM resolution in energy and momentum respectively, values are float, in eV, 1/A
        return:
            N/A
        '''
        self.dE = resdict['E']/np.sqrt(8*np.log(2)) #energy resolution FWHM
        self.dk = resdict['k']/np.sqrt(8*np.log(2)) #momentum resolution FWHM
        
    
    
###############################################################################    
###############################################################################    
##################  MAIN MATRIX ELEMENT EVALUATION  ###########################
###############################################################################
############################################################################### 
    
    def datacube(self,ARPES_dict):
        '''
            This function computes the photoemission matrix elements.
            Given a kmesh to calculate the photoemission over, the mesh is reshaped to an nx3 array and the Hamiltonian
            diagonalized over this set of k points. The matrix elements are then calculated for each 
            of these E-k points. The user can specify to do several things here. The fastest option is to set a 
            'slice' flag to True and select a binding energy of interest. Then a single energy is plotted.
            Alternatively, if no slice is selected, a series of text files can be generated which are then exported
            to for example Igor where they can be loaded like experimental data.
            args: ARPES_dict--experimental configuration:  c.f. docstring for class experiment
            for required key-value pairs in the ARPES_dict
        '''      
#        if ARPES_dict is not None:
#            self.update_pars(ARPES_dict)

        self.basis = self.rot_basis()

        print('Initiate diagonalization: ')
        self.diagonalize()
        print('Diagonalization Complete.')
        nstates = len(self.basis)
        if self.truncate:

            self.basis,self.Ev = self.truncate_model()
            

        dE = (self.cube[2][1]-self.cube[2][0])/self.cube[2][2]            
        dig_range = (self.cube[2][0]-5*dE,self.cube[2][1]+5*dE)
        
        self.cube_indx = np.array([[i,self.Eb[i]] for i in range(len(self.Eb)) if dig_range[0]<=self.Eb[i]<=dig_range[-1]])

           
        self.Mk = np.zeros((len(self.cube_indx),2,3),dtype=complex)
        self.pks = np.array([np.floor(np.floor(self.cube_indx[:,0]/nstates)/np.shape(self.X)[1]),np.floor(self.cube_indx[:,0]/nstates)%np.shape(self.X)[1],self.cube_indx[:,1]]).T
        kn = (2.*me/hb**2*(self.hv+self.cube_indx[:,1]-self.W)*q)**0.5*A
        self.th = np.array([np.arccos((kn[i]**2-self.X[int(self.pks[i,0]),int(self.pks[i,1])]**2-self.Y[int(self.pks[i,0]),int(self.pks[i,1])]**2)**0.5/kn[i]) if (kn[i]**2-self.X[int(self.pks[i,0]),int(self.pks[i,1])]**2-self.Y[int(self.pks[i,0]),int(self.pks[i,1])]**2)>=0 else -1 for i in range(len(self.cube_indx))])

        
        self.prefactors = np.array([o.sigma*np.exp((-1./self.mfp+1.0j*self.kz)*abs(o.depth)) for o in self.basis])
        self.Largs,self.Margs,Gmats,self.orbital_pointers = all_Y(self.basis) 
        self.Gbasis = Gmats[self.orbital_pointers]
        self.proj_arr = projection_map(self.basis)
        self.Bfuncs,self.radint_pointers = radint_lib.make_radint_pointer(ARPES_dict,self.basis,dig_range)


        print('Begin computing matrix elements: ')
        
        valid_indices = np.array([i for i in range(len(self.cube_indx)) if (self.th[i]>=0 and self.cube[2][0]<=self.cube_indx[i][1]<=self.cube[2][1])])
        
        if 'threads' in ARPES_dict.keys():
            self.thread_Mk(ARPES_dict['threads'],valid_indices)
        else:
            self.serial_Mk(valid_indices)

        print('\nDone matrix elements')


        return True
    
    
    
    
    
    def M_compute(self,i):
        '''
        The core method called during matrix element computation.
        args:
            i -- index and energy of state
        return Mtmp
        numpy array (2x3) of complex float corresponding to the matrix element projection for dm = -1,0,1 (columns) and spin down or up (rows)
        for a given state in k and energy
            
        '''
        nstates = len(self.TB.basis)
        phi = self.ph[int(self.cube_indx[i,0]/nstates)]
        th = self.th[i]
        Ylm_calls = Yvect(self.Largs,self.Margs,th,phi)[self.orbital_pointers]
        
        Mtmp = np.zeros((2,3),dtype=complex)
        B_eval = np.array([[b[0](self.cube_indx[i,1]),b[1](self.cube_indx[i,1])] for b in self.Bfuncs])
        pref = np.einsum('i,ij->ij',np.einsum('i,i->i',self.prefactors,self.Ev[int(self.cube_indx[i,0]/nstates),:,int(self.cube_indx[i,0]%nstates)]),B_eval[self.radint_pointers])  
        
        Gtmp = np.einsum('ij,ijkl->ikl',self.proj_arr,np.einsum('ijkl,ijkl->ijkl',Ylm_calls,self.Gbasis))

        if self.spin:
            Mtmp[0,:] = np.einsum('ij,ijk->k',pref[:int(len(self.basis)/2)],Gtmp[:int(len(self.basis)/2)])
            Mtmp[1,:] = np.einsum('ij,ijk->k',pref[int(len(self.basis)/2):],Gtmp[int(len(self.basis)/2):])
        else:
            Mtmp[0,:] = np.einsum('ij,ijk->k',pref,Gtmp)
                   
        return Mtmp
    
    
    
    
    def serial_Mk(self,indices):
        '''
        Run matrix element on a single thread
        args:
            indices -- list of all state indices for execution -- restricting states in cube_indx to those within the desired window
        return:
            None, directly modify the parent's Mk attribute
        '''
        for ii in indices:
            sys.stdout.write('\r'+progress_bar(ii+1,len(self.cube_indx)))
            self.Mk[ii,:,:]+=self.M_compute(ii)
        
    def thread_Mk(self,N,indices):
        '''
        Run matrix element on N threads using multiprocess functions
        NOTE 21/2/2019 -- this has not been optimized to show any measureable improvement over serial execution. May require a more clever
        way to do this to get a less disappointing result
        args:
            N -- number of threads
            indices -- list of all state indices for execution -- restricting states in cube_indx to those within the desired window
        return:
            None, directly modify the parent's Mk attribute
        '''
        div = int(len(indices)/N)
        pool = ThreadPool(N)
        results = np.array(pool.map(self.Mk_wrapper,[indices[ii*div:(ii+1)*div] for ii in range(N)]))
        pool.close()
        pool.join()
        results = results.reshape(len(indices),2,3)
        self.Mk[indices] = results
        
        
    def Mk_wrapper(self,ilist):
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
            
                
        
        
        
#    def SE_gen(self,SE_args,w,k):
#        '''
#        Define a self-energy function with which the spectral function
#        can be simulated. The user has a few options for how to pass this function: the self-
#        energy can be passed as an executable, a dictionary, a polynomial (passed as a
#        list of factors for each power in w), or as a constant float (imaginary part only).
#        Note that a k-dependent self-energy can be passed, but only as an executable.
#        args:
#            SE_args
#        
#        '''
#        
#        
#        if callable(SE_args):
#            try:
#                SE = SE_args(k,w)
#            except TypeError:
#                print('Using k-independent self-energy.')
#                try:
#                    SE = SE_args(w)
#                except TypeError:
#                    print('ERROR: invalid self-energy input!. Returning constant self energy 0.01i.')
#                    SE = -0.01*np.ones(len(w))
#        elif type(SE_args)==dict:
#            SE = gen_SE_KK(w,SE_args)
#        elif type(SE_args)==list:
#            SE = -1.0j*abs(poly(w,SE_args))#-1.0j*abs(poly(self.pks[:,2],ARPES_dict['SE'])) #absolute value mandates that self energy be particle-hole symmetric, i.e. SE*(k,w) = -SE(k,-w). Here we define the imaginary part explicitly only!
#        elif type(SE_args)==float:
#            SE = -1.0j*abs(SE_args*np.ones(len(w)))#-1.0j*abs(ARPES_dict['SE']*np.ones(len(self.pks[:,2])))
#        else:
#            SE = -0.01j*np.ones(len(w))#-0.01j*np.ones(len(self.pks[:,2]))
#            
#        return SE
        
    def spectral(self,ARPES_dict,slice_select=None):
        
        '''
        Take the matrix elements and build a simulated ARPES spectrum. 
        The user has several options here for the self-energy to be used: 
        most simply, nothing, then the imaginary part is taken to be -10meV to 
        give finite width, and the real part to 0, no renormalization.
        The next level is to pass a float different from -10 meV. Next is a polynomial
        which goes in as a list of coefficients [a_0,a_1,a_2...]. Finally is a dictionary,
        detailed below in gen_SE--this takes the input Im[Σ(k,w)] and uses Kramers-Kronig to
        build a consistent Re[Σ(k,w)]. This option WILL shift the band positions!!! Be WARNED!
        The ARPES_dict also constains a 'pol'-arization vector in Cartesian coordinates (lab frame).
        Also has option of a 'spin' projection, coming in the form of [+/-1, np.array([x,y,z])].
        'T' can be turned on or off [Bool, float] and set to some value in Kelvin. 'resolution'
        can also be updated.***SEE docstring for class experiment above for further details on ARPES_dict
        If slice_select is passed (list/tuple/float of length 2 (axis, index)) to force plotting.
        
        return: I, Ig the numpy array intensity maps and its resolution-broadened partner. Gaussian
        resolution broadening is the last operation performed, to be consistent with the practical experiment.
        '''
        
        self.update_resolution(ARPES_dict['resolution'])
        self.T = ARPES_dict['T']
        
        w = np.linspace(*self.cube[2])
        pol = pol_2_sph(ARPES_dict['pol'])
        
        if 'spin' in ARPES_dict.keys() and ARPES_dict['spin'] is not None:
            try:              
                sv = ARPES_dict['spin'][1]/np.linalg.norm(ARPES_dict['spin'][1])
            except KeyError:
                print('ERROR: Invalid spin-entry. See documentation for ARPES_lib.experiment')
                return None
            th = np.arccos(sv[2])
            ph = np.arctan2(sv[1],sv[0])
            if abs(self.ang)>0:
                ph+=self.ang
            Smat = np.array([[np.cos(th/2),np.exp(-1.0j*ph)*np.sin(th/2)],[np.sin(th/2),-np.exp(-1.0j*ph)*np.cos(th/2)]])
            Mspin = np.swapaxes(np.dot(Smat,self.Mk),0,1)
        
#        try:
        SE = self.SE_gen()
        
#        except KeyError:
#            
            

        if self.T[0]:
            fermi = vf(w/(kb*self.T[1]/q))
        else:
            fermi = np.ones(self.cube[2][2])
        I = np.zeros((self.cube[1][2],self.cube[0][2],self.cube[2][2]))
        if np.shape(SE)==np.shape(I):
            SE_k = True
        else:
            SE_k = False
        
        
        
        
        if 'spin' not in ARPES_dict.keys() or ARPES_dict['spin'] is None:
            for p in range(len(self.pks)):
                if abs(self.Mk[p]).max()>0:
                    I[int(np.real(self.pks[p,0])),int(np.real(self.pks[p,1])),:]+= (abs(np.dot(self.Mk[p,0,:],pol))**2 + abs(np.dot(self.Mk[p,1,:],pol))**2)*np.imag(-1./(np.pi*(w-self.pks[p,2]-(SE-0.0005j))))*fermi #changed SE[p] to SE
        else:
            
            for p in range(len(self.pks)):
                if abs(Mspin[p]).max()>0:
                    I[int(np.real(self.pks[p,0])),int(np.real(self.pks[p,1])),:]+= abs(np.dot(Mspin[p,int((ARPES_dict['spin'][0]+1)/2),:],pol))**2*np.imag(-1./(np.pi*(w-self.pks[p,2]-(SE-0.0005j))))*fermi #changed SE[p] to SE
        
        kxg = (self.cube[0][2]*self.dk/(self.cube[0][1]-self.cube[0][0]) if abs(self.cube[0][1]-self.cube[0][0])>0 else 0)
        kyg = (self.cube[1][2]*self.dk/(self.cube[1][1]-self.cube[1][0]) if abs(self.cube[1][1]-self.cube[1][0])>0 else 0)
        wg = (self.cube[2][2]*self.dE/(self.cube[2][1]-self.cube[2][0]) if abs(self.cube[2][1]-self.cube[2][0])>0 else 0)
        Ig = nd.gaussian_filter(I,(kxg,kyg,wg))
        
        if slice_select!=None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            if slice_select[0]==2: #FIXED ENERGY
                X,Y = np.meshgrid(np.linspace(*self.cube[0]),np.linspace(*self.cube[1]))
                p = ax.pcolormesh(X,Y,Ig[:,:,slice_select[1]],cmap=cm.magma)  
                plt.axis([self.cube[0][0],self.cube[0][1],self.cube[1][0],self.cube[1][1]])
            elif slice_select[0]==1: #FIXED KY
                X,Y = np.meshgrid(np.linspace(*self.cube[2]),np.linspace(*self.cube[0]))
                p = ax.pcolormesh(X,Y,Ig[slice_select[1],:,:],cmap=cm.magma)   
                plt.axis([self.cube[2][0],self.cube[2][1],self.cube[0][0],self.cube[0][1]])
            elif slice_select[0]==0: # FIXED KX
                X,Y = np.meshgrid(np.linspace(*self.cube[2]),np.linspace(*self.cube[1]))
                p = ax.pcolormesh(X,Y,Ig[:,slice_select[1],:],cmap=cm.magma)    
                plt.axis([self.cube[2][0],self.cube[2][1],self.cube[1][0],self.cube[1][1]])
                
            plt.colorbar(p,ax=ax)
        
        return I,Ig
    
    def plot_gui(self,ARPES_dict):
        TK_win = Tk_plot.plot_intensity_interface(self,ARPES_dict)
        return TK_win
        
        
        
###############################################################################    
###############################################################################    
################### WRITE ARPES MAP TO FILE ###################################
###############################################################################
###############################################################################        
     
    def write_map(self,_map,directory):
        '''
        Write the intensity map to a text file in the indicated directory.
        args:
            _map -- numpy array of float to write
            directory -- string, name of directory + the file-lead name e.g. /Users/name/ARPES_maps/room_temp_superconductor'
            will produce a series of files labeled as room_temp_superconductor_xx.txt in the /Users/name/ARPES_maps/ subfolder
        '''
        for i in range(np.shape(_map)[2]):   
            filename = directory + '_{:d}.txt'.format(i)
            self.write_Ik(filename,_map[:,:,i])
        return True

    def write_params(self,Adict,parfile):
        '''
        Generate metadata text file  associated with the saved map.
        args:
            Adict -- ARPES_dict same as in above functions, containing relevant experimental parameters
            parfile -- destination for the metadata (string)
        
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
        Sub-function for producing the textfiles associated with a 2dimensional numpy array of float
        args:
            filename -- string indicating destination of file
            mat -- 2 dimensional numpy array of float
        
        '''
        with open(filename,"w") as destination:
            for i in range(np.shape(mat)[0]):
                tmpline = " ".join(map(str,mat[i,:]))
                tmpline+="\n"
                destination.write(tmpline)
        destination.close()
        return True
    
    def rot_basis(self):
        '''
        Rotate the basis orbitals and their positions in the lab frame to be consistent with the
        experimental geometry
        return:
            rotated copy of the basis (list of orbital objects) if the rotation is non-zero
            otherwise, just return the untouched basis
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
    
    
    

    def spin_rot(self,spin_ax):
        '''
        For spin-ARPES, want to project the eigenstates onto different spin axes than simply the canonical z axis. 
        To do so we perform an unitary transformation of the eigenvectors. 
        A generic spin vector basis can be defined as:
            |+n> = cos(theta/2)|+z> + e(i*phi)*sin(theta/2)|-z>
            |-n> = sin(theta/2)|+z> - e(i*phi)*cos(theta/2)|-z>
        Then we define a
         
            |  <-n|-z> , <-n|+z> |
        U = |                    |
            |  <+n|-z> , <+n|+z> |
        
        Then using a Kroenecker (tensor) product, we form a len(basis)xlen(basis) square matrix which can be separated
        into 4 blocks defined by the above 4x4. Where each block is uniform
        
        If the ARPES calculation is being done over a rotated k-space, we need to rotate our definitions of the spin
        basis as well, and so the phi values in the 'ax_dict' will be increased by the angle by which we are rotating the
        k-space
        args: spin_ax -- numpy array indicating the vector direction of the spin projection (numpy array of 3 float)
        return spin_mat --numpy array complex of size NxN with N the basis length
        '''
        
        if spin_ax is not None:
            spin_mat = np.zeros((len(self.TB.basis),len(self.TB.basis)))
            spin_ax = spin_ax/np.linalg.norm(spin_ax)
            th = np.arccos(spin_ax[2])
            ph = np.arctan2(spin_ax[1],spin_ax[0])
            if abs(self.ang)>0:
                print(self.ang)
                ph -=self.ang
            US = np.array([[-np.cos(th/2)*np.exp(-1.0j*ph),np.sin(th/2)],[np.sin(th/2)*np.exp(-1.0j*ph),np.cos(th/2)]])
            print(US)
            
            spin_mat = np.kron(US,np.identity(int(len(self.TB.basis)/2)))
    
            return spin_mat
        else:
            return np.identity(len(self.TB.basis))
    



###############################################################################    
###############################################################################    
######################## SUPPORT FUNCTIONS#####################################
###############################################################################
###############################################################################
        
        
def con_ferm(x):      ##Typical energy scales involved and temperatures involved give overflow in exponential--define a new fermi function that works around this problem
    tmp = 0.0
    if x<709:
        tmp = 1.0/(np.exp(x)+1)
    return tmp


vf = np.vectorize(con_ferm)

def pol_Y(pvec):
    '''
    Transform polarization vector from cartesian to spherical components
    args: pvec -- len(3) numpy array with form ([p_x,p_y,p_z])
    return: len(3) numpy array describing polarization vector projected as ([p_1,p_0,p_-1])
    '''
    R = np.array([[-np.sqrt(0.5),-np.sqrt(0.5)*1.0j,0],[0,0,1],[np.sqrt(0.5),-np.sqrt(0.5)*1.0j,0]])
    return np.dot(R,pvec)

def base2sph(basis):
    '''
    Define a basis transformation from user to Y_lm, this will simplify the calculation of matrix elements significantly
    args: basis --list of orbital objects
    '''
    o2l = []
    n,l,pos,s = basis[0].n,basis[0].l,basis[0].pos,basis[0].spin
    mvals =[m for m in range(-l,l+1)]
    tmp = np.zeros((2*l+1,len(basis)),dtype=complex)
    for b in basis:
        if b.n!=n or b.l!=l or np.linalg.norm(b.pos-pos)>0.0 or b.spin!=s:
            for t in tmp:
                o2l.append(t)
            tmp = np.zeros((2*b.l+1,len(basis)),dtype=complex)
            n,l,pos,s = b.n,b.l,b.pos,b.spin
            mvals = mvals + [m for m in range(-l,l+1)]

            
            
        for m in b.proj:
            tmp[int(b.l+m[-1]),int(b.index)] +=m[0]+1.0j*m[1]
    for t in tmp:
        o2l.append(t)
        
    return np.array(o2l),np.array(mvals)
            
    
def pol_2_sph(pol):
    '''
    return pol vector in spherical harmonics -- order being Y_11, Y_10, Y_1-1
    '''
    M = np.sqrt(0.5)*np.array([[-1,1.0j,0],[0,0,np.sqrt(2)],[1.,1.0j,0]])
    return np.dot(M,pol)

    
def lambda_gen(val):
    return lambda x: val


def poly(x,args):
    '''
    Recursive polynomial function.
    args:
        x -- input value at which to evaluate the polynomial (float, int or numpy array of numeric type)
        args -- list of coefficients, in INCREASING polynomial order i.e. [a_0,a_1,a_2] for y = a_0 + a_1 * x + a_2 *x **2
    return:
        polynomial evaluated at x, same datatype as input (or at worst int -> float)
    '''
    if len(args)==0:
        return 0
    else:
        return x**(len(args)-1)*args[-1] + poly(x,args[:-1])
    
  
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
#        
        
    
def progress_bar(N,Nmax):
    frac = N/Nmax
    st = ''.join(['|' for i in range(int(frac*30))])
    st = '{:30s}'.format(st)+'{:3d}%'.format(int(frac*100))
    return st


def find_max_dE(Eb):
    dE_max = abs(np.subtract(Eb[1:,:],Eb[:-1,:])).mean()
    return dE_max 
            

###############################################################################    
###############################################################################    
######################## ANGULAR INTEGRALS ####################################
###############################################################################
###############################################################################
        
def G_dic():
    '''
    Initialize the gaunt coefficients associated with all possible transitions relevant
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
    args:
        basis -- list of orbital objects
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
    
    maxproj = max([len(o.proj) for o in basis])
    projarr = np.zeros((len(basis),maxproj),dtype=complex)
    for ii in range(len(basis)):
        for pj in range(len(basis[ii].proj)):
            proj = basis[ii].proj[pj]
            projarr[ii,pj] = proj[0]+1.0j*proj[1]
    return projarr#,lm




Yvect = np.vectorize(Ylm.Y,otypes=[complex])

def Gmat_make(lm,Gdictionary):
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
    
    