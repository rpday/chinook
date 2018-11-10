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
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import ubc_tbarpes.klib as K_lib
import ubc_tbarpes.orbital as olib
import ubc_tbarpes.adaptive_int as adint
from scipy.interpolate import interp1d
import ubc_tbarpes.Ylm as Ylm 
import scipy.ndimage as nd
import ubc_tbarpes.Tk_plot as Tk_plot
from scipy.signal import hilbert


####PHYSICAL CONSTANTS RELEVANT TO CALCULATION#######
hb = 6.626*10**-34/(2*np.pi)
c  = 3.0*10**8
q = 1.602*10**-19
A = 10.0**-10
me = 9.11*10**-31
mN = 1.67*10**-27
kb = 1.38*10**-23
Lpd = {0:[1],1:[0,2],2:[1,3],3:[2,4]}




###
class experiment:
    '''
    The experiment object is at the centre of the ARPES matrix element calculation.
    This object keeps track of the experimental geometry as well as a local copy of the tight-binding model
    and its dependents. Such a copy is used to avoid corruption of these objects in the global space during a given run
    of the ARPES experiment.
    
    args: TB -- instance of a tight-binding model object
        ARPES_dict -- dictionary of relevant experimental parameters including:
            'hv': photon energy (float,eV), 'mfp': mean-free path (float,Ang.),
            'dE': energy resolution (float, FWHM), 'dk': momentum resolution (float, 1/Ang)
            'ang': rotation of sample about normal emission (z) axis (float, radian),
            'T': Temperature of sample (float, Kelvin)
            'W': work function (float, eV)
            'cube': momentum and energy domain (dictionary: 'kz' is float, all others are list or tuple of floats Xo,Xf,dX)
        if running a slab calculation, the eigenvectors are truncated below a certain depth
    
    '''
    def __init__(self,TB,ARPES_dict):#hv,pol,mfp,dE,dk,T,ang=0.0,W=4.0):
        self.TB = TB
        self.hv = ARPES_dict['hv']
        self.mfp = ARPES_dict['mfp'] #photoelectron mean free path for escape
        self.dE = ARPES_dict['resolution']['E']/np.sqrt(8*np.log(2)) #energy resolution FWHM
        self.dk = ARPES_dict['resolution']['k']/np.sqrt(8*np.log(2)) #momentum resolution FWHM
        self.ang = ARPES_dict['angle']
        self.T = ARPES_dict['T']
        self.W = ARPES_dict['W']
        self.cube = (ARPES_dict['cube']['X'],ARPES_dict['cube']['Y'],ARPES_dict['cube']['E'])
        self.kz = ARPES_dict['cube']['kz']
        try:
            self.truncate = ARPES_dict['slab']
        except KeyError:
            self.truncate = False

    
    
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
        self.Eb = np.reshape(self.Eb,(np.shape(self.Eb)[-1]*np.shape(self.X)[0]*np.shape(self.X)[1])) 
        
        
        
    def truncate_model(self,local_basis):
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
        depths = np.array([abs(oi.depth) for oi in local_basis])
        i_start = np.where(depths<4*self.mfp)[0][0]

        tmp_basis = []
        #CASE 1: BASIS INCLUDES BOTH SPIN DOF
        if local_basis[0].spin!=local_basis[int(len(local_basis)/2)].spin:
            
            switch = (int(len(local_basis)/2))
            tmp_basis = local_basis[i_start:switch] + local_basis[(switch+i_start):]
            
            Evec = np.zeros((np.shape(self.Ev)[0],len(tmp_basis),np.shape(self.Ev)[-1]),dtype=complex)
            
            Evec[:,:(switch-i_start),:] =self.Ev[:,i_start:switch,:]
            Evec[:,(switch-i_start):,:] = self.Ev[:,(switch+i_start):,:]
        #CASE 2: BASIS IS SPINLESS
        else:
            
            tmp_basis = local_basis[i_start:]
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
        '''      
        

        tmp_basis = self.rot_basis()

        print('Initiate diagonalization: ')
        self.diagonalize()
        print('Diagonalization Complete.')
        nstates = len(tmp_basis)
        if self.truncate:

            tmp_basis,self.Ev = self.truncate_model(tmp_basis)
        dE = (self.cube[2][1]-self.cube[2][0])/self.cube[2][2]
                
        
        dig_range = np.arange(self.cube[2][0]-5*dE,self.cube[2][1]+5*dE,dE)
                
        cube_indx = np.array([[i,self.Eb[i]] for i in range(len(self.Eb)) if dig_range[0]<=self.Eb[i]<=dig_range[-1]])
        Gvals = G_dic()
        
        try:
            self.Bvals = {}
            for bkey in ARPES_dict['Brads']:
                self.Bvals[bkey] = lambda_gen(ARPES_dict['Brads'][bkey])
        except KeyError:
            self.Bvals = self.Bdic(dig_range,tmp_basis)
           
        self.pks = np.zeros((len(cube_indx),3),dtype=float)
        self.Mk = np.zeros((len(cube_indx),2,3),dtype=complex)
        self.pks = np.array([np.floor(np.floor(cube_indx[:,0]/nstates)/np.shape(self.X)[1]),np.floor(cube_indx[:,0]/nstates)%np.shape(self.X)[1],cube_indx[:,1]]).T

        kn = np.sqrt(2.*me/hb**2*(self.hv+self.Eb-self.W)*q)*A
        self.th = np.array([np.arccos(np.sqrt(kn[int(cube_indx[i,0])]**2-self.X[int(self.pks[i,0]),int(self.pks[i,1])]**2-self.Y[int(self.pks[i,0]),int(self.pks[i,1])]**2)/kn[int(cube_indx[i,0])]) for i in range(len(cube_indx))])
        nstates = len(self.TB.basis)


        
        tol = 0.01
        strmats = Gstrings()
        print('Begin computing matrix elements: ')
        for i in range(len(cube_indx)):
            if not ARPES_dict['slice'][0]:
                if self.cube[2][0]<=cube_indx[i][1]<=self.cube[2][1]:
                    tmp_M = self.M_compute(Gvals,self.Bvals,i,cube_indx[i],kn[i],tmp_basis,tol,strmats) ###
                else:
                    tmp_M=0.0
            elif abs(cube_indx[i][1]-ARPES_dict['slice'][1])<self.dE:
                tmp_M = self.M_compute(Gvals,self.Bvals,i,cube_indx[i],kn[i],tmp_basis,tol,strmats)*np.exp(-(cube_indx[i][1]-ARPES_dict['slice'][1])**2/(2*self.dE)) ####
            else:
                tmp_M = 0.0


            self.Mk[i,:,:] = tmp_M
        print('Done matrix elements')


        return True
    

    def M_compute(self,G,B,i,cube,kn,basis,tol,strmats):
        '''
        The core method called during matrix element computation.
        args:
            G -- dictionary of relevant Gaunt coefficients
            B -- dictionary of radial integral functions, to be evaluated at the designated energy
            
        '''
        nstates = len(self.TB.basis)
        phi = self.ph[int(cube[0]/nstates)]
        th = self.th[i]
        Mtmp = np.zeros((2,3),dtype=complex)
        
        psi = self.Ev[int(cube[0]/nstates),:,int(cube[0]%nstates)]
        
        for coeff in list(enumerate(psi)):

            if abs(coeff[1])>tol:
                o = basis[coeff[0]]
                pref = o.sigma*coeff[1]*np.exp((-1./self.mfp)*abs(o.depth))#*np.exp(1.0j*(-self.kz*o.pos[2]-self.X[int(self.pks[i,0]),int(self.pks[i,1])]*o.pos[0]-self.Y[int(self.pks[i,0]),int(self.pks[i,1])]*o.pos[1]))
                L = [lp for lp in ([o.l-1,o.l+1] if (o.l-1)>=0 else [o.l+1])]

                for lp in L:

                    tmp_B = B['{:d}-{:d}-{:d}-{:d}'.format(o.atom,o.n,o.l,lp)](cube[1])
                    tmp_G = np.zeros(3,dtype=complex)
                    for op in o.proj:
                        tmp_G += (op[0]+1.0j*op[1])*np.array([G['{:d}{:d}{:d}{:d}'.format(o.l,lp,int(op[-1]),-1)]*Ylm.Y(lp,int(op[-1])-1,th,phi),G['{:d}{:d}{:d}{:d}'.format(o.l,lp,int(op[-1]),0)]*Ylm.Y(lp,int(op[-1]),th,phi),G['{:d}{:d}{:d}{:d}'.format(o.l,lp,int(op[-1]),1)]*Ylm.Y(lp,int(op[-1])+1,th,phi)])

                    Mtmp[int((o.spin+1)/2),:]+=pref*tmp_B*tmp_G
                   
        return Mtmp



###############################################################################    
###############################################################################    
####################### DATA VIEWING  #########################################
###############################################################################
############################################################################### 
    
    def plot_slice(self,ARPES_dict):
        '''
        If a single slice has been selected for computing, then we can produce an image of this slice quickly
        '''
        pol = pol_2_sph(ARPES_dict['pol'])
        if ARPES_dict['slice'][0]:
            xv,yv = ARPES_dict['cube']['X'],ARPES_dict['cube']['Y']
            x = np.linspace(*xv)
            y = np.linspace(*yv)
            X,Y = np.meshgrid(x,y)
            I = np.zeros((np.shape(X)))
            
            for p in range(len(self.pks)):
                if abs(self.Mk[p].max())>0:
                    I[int(np.real(self.pks[p,0])),int(np.real(self.pks[p,1]))]+= abs(np.dot(self.Mk[p,0,:],pol))**2 + abs(np.dot(self.Mk[p,1,:],pol))**2
            kxg = self.dk/(x[1]-x[0])
            kyg = self.dk/(y[1]-y[0])
            Ig = nd.gaussian_filter(I,(kxg,kyg))
            fig = plt.figure()
            ax = fig.add_subplot(111)
            p = ax.pcolormesh(X,Y,Ig,cmap=cm.magma)
            plt.axis([x[0],x[-1],y[0],y[-1]])
            plt.colorbar(p,ax=ax)
            return Ig
        else:
            return None
        
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
        can also be updated.
        If slice_select is passed (list/tuple/float of length 2 (axis, index)) to force plotting.
        
        return: I, Ig the numpy array intensity maps and its resolution-broadened partner. Gaussian
        resolution broadening is the last operation performed, to be consistent with the practical experiment.
        '''
        
        self.update_resolution(ARPES_dict['resolution'])
        self.T = ARPES_dict['T']
        
        w = np.linspace(*self.cube[2])
        pol = pol_2_sph(ARPES_dict['pol'])
        
        if ARPES_dict['spin']!=None:
            sv = ARPES_dict['spin'][1]/np.linalg.norm(ARPES_dict['spin'][1])
            th = np.arccos(sv[2])
            ph = np.arctan2(sv[1],sv[0])
            if abs(self.ang)>0:
                ph+=self.ang
            Smat = np.array([[np.cos(th/2),np.exp(-1.0j*ph)*np.sin(th/2)],[np.sin(th/2),-np.exp(-1.0j*ph)*np.cos(th/2)]])
            Mspin = np.swapaxes(np.dot(Smat,self.Mk),0,1)
        if callable(ARPES_dict['SE']):
            SE = ARPES_dict['SE'](w)
        elif type(ARPES_dict['SE'])==dict:
            SE = gen_SE(w,ARPES_dict['SE'])#gen_SE(self.pks[:,2],ARPES_dict['SE'])
        elif type(ARPES_dict['SE'])==list:
            SE = -1.0j*abs(poly(w,ARPES_dict['SE']))#-1.0j*abs(poly(self.pks[:,2],ARPES_dict['SE'])) #absolute value mandates that self energy be particle-hole symmetric, i.e. SE*(k,w) = -SE(k,-w). Here we define the imaginary part explicitly only!
        elif type(ARPES_dict['SE'])==float:
            SE = -1.0j*abs(ARPES_dict['SE']*np.ones(len(w)))#-1.0j*abs(ARPES_dict['SE']*np.ones(len(self.pks[:,2])))
        else:
            SE = -0.01j*np.ones(len(w))#-0.01j*np.ones(len(self.pks[:,2]))
            

        if self.T[0]:
            fermi = vf(w/(kb*self.T[1]/q))
        else:
            fermi = np.ones(self.cube[2][2])
        I = np.zeros((self.cube[1][2],self.cube[0][2],self.cube[2][2]))
        
        
        
        
        if ARPES_dict['spin'] is None:
            for p in range(len(self.pks)):
                if abs(self.Mk[p]).max()>0:
                    I[int(np.real(self.pks[p,0])),int(np.real(self.pks[p,1])),:]+= (abs(np.dot(self.Mk[p,0,:],pol))**2 + abs(np.dot(self.Mk[p,1,:],pol))**2)*np.imag(-1./(np.pi*(w-self.pks[p,2]-(SE-0.0005j))))*fermi #changed SE[p] to SE
        else:
            for p in range(len(self.pks)):
                if abs(Mspin[p]).max()>0:
                    I[int(np.real(self.pks[p,0])),int(np.real(self.pks[p,1])),:]+= abs(np.dot(Mspin[p,int((ARPES_dict['spin'][0]+1)/2),:],pol))**2*np.imag(-1./(np.pi*(w-self.pks[p,2]-(SE-0.0005j))))*fermi #changed SE[p] to SE
        
        kxg = (self.cube[0][2]*self.dk/(self.cube[0][1]-self.cube[0][0]) if abs(self.cube[0][1]-self.cube[0][0])>0 else 0)
        kyg = (self.cube[1][2]*self.dk/(self.cube[1][1]-self.cube[1][0]) if abs(self.cube[1][1]-self.cube[1][0])>0 else 0)
        wg = (self.cube[2][2]*self.dE/(self.cube[2][1]-self.cube[2][0]) if abs(self.cube[2][1]-self.cube[2][0]) else 0)
        
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
    
    def plot_gui(self,Adict):
        TK_win = Tk_plot.plot_intensity_interface(self,Adict)
        
        
        
###############################################################################    
###############################################################################    
################### WRITE ARPES MAP TO FILE ###################################
###############################################################################
###############################################################################        
     
    def write_map(self,_map,directory):
        for i in range(np.shape(_map)[2]):   
            filename = directory + '_{:d}.txt'.format(i)
            self.write_Ik(filename,_map[:,:,i])
        return True

    def write_params(self,Adict,parfile):
        
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
        with open(filename,"w") as destination:
            for i in range(np.shape(mat)[0]):
                tmpline = " ".join(map(str,mat[i,:]))
                tmpline+="\n"
                destination.write(tmpline)
        destination.close()
        return True

###############################################################################    
###############################################################################    
######################## RADIAL INTEGRALS #####################################
###############################################################################
###############################################################################         


    def Brad_calc(self,k_norm,basis):
        '''
        Compute dictionary of radial integrals evaluated at a single |k| value for the whole basis.
        Will avoid redundant integrations by checking for the presence of an identical dictionary key.
        The integration is done as a simple adaptive integration algorithm, defined in the adaptive_int library
        args:
            k_norm -- float length of the k-vector (as an argument for the spherical Bessel Function)
            basis -- list of orbital objects, from which the orbital angular momentum is extracted
        returns:
            dictionary of key value pairs in form -- 'ATOM-N-L':Bval
        '''
        Bdic = {}
        for o in basis:
            tmp='{:d}-{:d}-{:d}'.format(o.atom,o.n,o.l)
            L=[x for x in [o.l-1,o.l+1] if x>=0]
            for lp in L:
                Blabel=tmp+'-'+str(lp)
                try:
                    Bdic[Blabel]
                    continue
                except KeyError:
                    trueconverge = False
                    rmax = 10.0
                    while not trueconverge:
                        tmp_B = adint.Bintegral(0.001,rmax,10.0**-10,lp,k_norm,int(o.Z),o.label)
                        if abs(tmp_B)<10**-10:
                            rmax/=2.0
                        else:
                            trueconverge = True                        
                    Bdic[Blabel]=tmp_B
        return Bdic

    def Bdic(self,Eb,basis):
        '''
        Function for computing dictionary of radial integrals. Can pass either an array of
        binding energies or a single binding energy as a float. In either case, returns a dictionary
        however the difference being that the key value pairs will have a value which is itself either a
        float, or an interpolation mesh over the range of the binding energy array.
        The output can then be used by either writing Bdic['key'] or Bdic['key'](valid float between endpoints of input array)
        args:
            Eb -- float or numpy array
            basis -- orbital basis, passed to the B
        '''
        if type(Eb)==float:
            kval = np.sqrt(2.0*me/hb**2*((self.hv-self.W)+Eb)*q)*A 
            return (self.Brad_calc(kval,basis) if ((self.hv-self.W)+Eb)>=0 else 0)
        elif type(Eb)==np.ndarray:
            Brad_es=np.linspace(Eb[0],Eb[-1],5)
            BD_coarse={}
            for en in Brad_es:
                k_coarse = np.sqrt(2.0*me/hb**2*((self.hv-self.W)+en)*q)*A #calculate full 3-D k vector at this surface k-point given the incident radiation wavelength, and the energy eigenvalue, note binding energy follows opposite sign convention
                tmp_Bdic = (self.Brad_calc(k_coarse,basis) if ((self.hv-self.W)+en)>=0 else 0)
                for b in tmp_Bdic:
                    try:
                        BD_coarse[b].append(tmp_Bdic[b])
                    except KeyError:
                        BD_coarse[b] = [tmp_Bdic[b]]
    
            Brad = {}     
            for b in BD_coarse:
                f = interp1d(Brad_es,BD_coarse[b],kind='cubic')
                Brad[b] = f
            return Brad
        else:
            print('Invalid binding energy type--enter float or numpy array!')
            return False



    def Bdic_PP(self,kn):
        Brads = {}
        for p in self.PP_WF:
            Bm,Bp = self.PP_WF[p].pseudo_integral(kn)
            Brads[p+str(int(p[-1])-1)] = Bm
            Brads[p+str(int(p[-1])+1)] = Bp
        Blist = [{PP:Brads[PP][i] for PP in Brads} for i in range(len(kn))]
        return Blist
    
###############################################################################    
###############################################################################    
######################### BASIS ROTATIONS #####################################
###############################################################################
###############################################################################    
    
    
    def rot_basis(self):
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
#            ax_dict = {'x':[np.pi/2,0.0],'y':[np.pi/2,np.pi/2],'z':[0,0]}
            if abs(self.ang)>0:
                print(self.ang)
                ph -=self.ang
#                ax_dict[spin_ax][1]-=self.ang
            US = np.array([[-np.cos(th/2)*np.exp(-1.0j*ph),np.sin(th/2)],[np.sin(th/2)*np.exp(-1.0j*ph),np.cos(th/2)]])
#            US = np.array([[-np.cos(ax_dict[spin_ax][0]/2)*np.exp(-1.0j*ax_dict[spin_ax][1]),np.sin(ax_dict[spin_ax][0]/2)],[np.sin(ax_dict[spin_ax][0]/2)*np.exp(-1.0j*ax_dict[spin_ax][1]),np.cos(ax_dict[spin_ax][0]/2)]])
            print(US)
            
            spin_mat = np.kron(US,np.identity(int(len(self.TB.basis)/2)))
    
            return spin_mat
        else:
            return np.identity(len(self.TB.basis))
    
###############################################################################    
###############################################################################    
######################## ANGULAR INTEGRALS ####################################
###############################################################################
###############################################################################
        
def G_dic():
    
    llp = [[l,lp] for l in range(4) for lp in ([l-1,l+1] if (l-1)>=0 else [l+1])]    

    llpmu = [[l[0],l[1],m,u] for l in llp for m in np.arange(-l[0],l[0]+1,1) for u in [-1,0,1]]
    keyvals = [[str(l[0])+str(l[1])+str(l[2])+str(l[3]), Ylm.gaunt(l[0],l[2],l[1]-l[0],l[3])] for l in llpmu]
    G_dict = dict(keyvals)
    
    for gi in G_dict:
        if np.isnan(G_dict[gi]):
            G_dict[gi]=0.0       
    return G_dict

def Gstrings():
    strmats = {}
    for l in Lpd:
        strmats[l] = np.array([[str(l),str(lp),str(m),str(u)] for u in [-1,0,1] for lp in Lpd[l] for m in np.arange(-l,l+1,1)])
#        strmats[l] = np.reshape(llpmu,(3,len(Lpd[l]),2*l+1))
    
    return strmats


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

def tstat(t,ts):
    if t<ts[0]:
        ts[0]=t
    if t>ts[1]:
        ts[1]=t
    ts[2]+=1
    ts[3]+=t
    
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
    
    
#def abs_poly(x,args):
#    '''
#    
#    '''
#    if len(args)==0:
#        return 0
#    else:
#        return abs(x)**(len(args)-1)*abs(args[-1]) + abs_poly(x,args[:-1])
    
    
def gen_SE(w,SE_args):
    '''
    The total self-energy is computed using Kramers' Kronig relations:
        
        The user can pass the self-energy in the form of either a callable function, a list of polynomial coefficients, or as a numpy array with shape Nx2 (with the first
        column an array of frequency values, and the second the values of a function). For the latter option, the user is responsible for ensuring that the function goes 
        to zero at the tails of the domain. In the former two cases, the 'cut' parameter is used to impose an exponential cutoff near the edge of the domain to ensure this 
        is the case. In all cases the input imaginary self-energy must be single-signed to ensure it is purely even function. It is forced to be negative in all cases to give
        a positive spectral function.
        With the input defined, along with the energy range of interest to the calculation, a MUCH larger domain (100x in the maximal extent of the energy region of interest) is defined
        wf. This is the domain over which we evaluate the Hilbert transform, which itself is carried out using:
        the scipy.signal.hilbert() function. This function acting on an array f: H(f(x)) -> f(x) + i Hf(x). It relies on the FFT performed on the product of the sgn(w) and F(w) functions, and then IFFT back
        so that we can use this to extract the real part of the self energy, given only the input.
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
        wlim = abs(w).max()
        wf = np.linspace(-100*wlim,100*wlim,5000)
        if type(SE_args['imfunc'])==np.ndarray and np.shape(SE_args['imfunc'])[1]==2:
            wf = SE_args['imfunc'][:,0]
            imSE = SE_args['imfunc'][:,1]
        else:
            if callable(SE_args['imfunc']):
                imSE = SE_args['imfunc'](wf)
                if np.real(imSE).max()==0.0:
                    print('Input WARNING: The imaginary part of the self-energy should be passed as real-valued function (i.e. suppress the 1.0j factor). Taking imaginary part as real and proceeding.')
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
            
            
    
    
    
    
    
    #    def matel_matrix(self,B,G,basis):
#        
#        B_funcs= [[B['{:d}-{:d}-{:d}-{:d}'.format(o.atom,o.n,o.l,o.l-1)],B['{:d}-{:d}-{:d}-{:d}'.format(o.atom,o.n,o.l,o.l+1)]] for o in basis]
#        def GBmats(E,theta,phi):
#            mv1 = np.array([[0*B_funcs[i][0](E)] for i in range(len(B_funcs))])
#            mv2 = np.array([[0*B_funcs[i][0](E)] for i in range(len(B_funcs))])
#            mv3 = np.array([[0*B_funcs[i][0](E)] for i in range(len(B_funcs))])
#            for i in range(len(mv1)):
#                for op in basis[i].proj:
#                    mv1[i] += B_funcs[i][0](E)*(op[0]+1.0j*op[1])*G['{:d}{:d}{:d}{:d}'.format(basis[i].l,basis[i].l-1,int(op[-1]),-1)]*Ylm.Y(basis[i].l-1,int(op[-1])-1,theta,phi)+ B_funcs[i][1](E)*(op[0]+1.0j*op[1])*G['{:d}{:d}{:d}{:d}'.format(basis[i].l,basis[i].l+1,int(op[-1]),-1)]*Ylm.Y(basis[i].l+1,int(op[-1])-1,theta,phi)
#                    mv2[i] += B_funcs[i][0](E)*(op[0]+1.0j*op[1])*G['{:d}{:d}{:d}{:d}'.format(basis[i].l,basis[i].l-1,int(op[-1]),0)]*Ylm.Y(basis[i].l-1,int(op[-1]),theta,phi)+B_funcs[i][1](E)*(op[0]+1.0j*op[1])*G['{:d}{:d}{:d}{:d}'.format(basis[i].l,basis[i].l+1,int(op[-1]),0)]*Ylm.Y(basis[i].l+1,int(op[-1]),theta,phi)
#                    mv3[i] += basis[i].sigma*(B_funcs[i][0](E)*(op[0]+1.0j*op[1])*G['{:d}{:d}{:d}{:d}'.format(basis[i].l,basis[i].l-1,int(op[-1]),1)]*Ylm.Y(basis[i].l-1,int(op[-1])+1,theta,phi)+ B_funcs[i][1](E)*(op[0]+1.0j*op[1])*G['{:d}{:d}{:d}{:d}'.format(basis[i].l,basis[i].l+1,int(op[-1]),1)]*Ylm.Y(basis[i].l+1,int(op[-1])+1,theta,phi))
#
#            return np.array([mv1,mv2,mv3])
#
#        
#        return GBmats#mat_m1,mat_0,mat_p1
    
    
#    def M_compute_2(self,GBval,cube,basis,spinmat,div):
#
#        Mtmp = np.zeros((2,3),dtype=complex)
#        
#        psi = self.Ev[int(cube[0]/len(basis)),:,int(cube[0]%len(basis))]
#                
#        if spinmat is not None:
#            psi = np.dot(spinmat,psi)
#        Mtmp[:,0] = np.array([np.dot(GBval[0][:div,0],psi[:div]),np.dot(GBval[0][div:,0],psi[div:])])
#        Mtmp[:,1] = np.array([np.dot(GBval[1][:div,0],psi[:div]),np.dot(GBval[1][div:,0],psi[div:])])
#        Mtmp[:,2] = np.array([np.dot(GBval[2][:div,0],psi[:div]),np.dot(GBval[2][div:,0],psi[div:])])
#        
#            
#    