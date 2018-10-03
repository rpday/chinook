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
import ubc_tbarpes.adaptive_int as adint
from scipy.interpolate import interp1d
import ubc_tbarpes.Ylm as Ylm 
import scipy.ndimage as nd
import ubc_tbarpes.Tk_plot as Tk_plot



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
        
        
        
    def truncate_model(self):
        '''
        For slab calculations, the number of basis states becomes a significant memory load, as well as a time bottleneck.
        In reality, an ARPES calculation only needs the small number of basis states near the surface. Then for slab-calculations,
        we can truncate the basis and eigenvectors used in the calculation to dramatically improve our capacity to perform such calculations
        We keep all eigenvectors, but retain only the projection of the basis states within 2*the mean free path of the surface. The states associated
        with this projection are retained, while remainders are not
        
        '''
        depths = np.array([abs(oi.depth) for oi in self.TB.basis])
        i_start = np.where(depths<2*self.mfp)[0][0]

        tmp_basis = []
        #CASE 1: BASIS INCLUDES BOTH SPIN DOF
        if self.TB.basis[0].spin!=self.TB.basis[int(len(self.TB.basis)/2)].spin:
            
            switch = (int(len(self.TB.basis)/2))
            tmp_basis = self.TB.basis[i_start:switch] + self.TB.basis[(switch+i_start):]
            
            Evec = np.zeros((np.shape(self.Ev)[0],len(tmp_basis),np.shape(self.Ev)[-1]),dtype=complex)
            
            Evec[:,:(switch-i_start),:] =self.Ev[:,i_start:switch,:]
            Evec[:,(switch-i_start):,:] = self.Ev[:,(switch+i_start):,:]
        #CASE 2: BASIS IS SPINLESS
        else:
            
            tmp_basis = self.TB.basis[i_start:]
            Evec=self.Ev[:,i_start:,:]
        return tmp_basis,Evec
    
    def update_resolution(self,resdict):
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

            tmp_basis,self.Ev = self.truncate_model()
        dE = (self.cube[2][1]-self.cube[2][0])/self.cube[2][2]
        
        dig_range = np.arange(self.Eb.min()-5*dE,self.Eb.max()+dE*5,dE)
        
        dig_range = np.arange(self.cube[2][0]-5*dE,self.cube[2][1]+5*dE,dE)
                
        ##cube_indx is the position in the 3d cube of K and energy where this band is
        cube_indx = np.array([[i,self.Eb[i]] for i in range(len(self.Eb)) if dig_range[0]<=self.Eb[i]<=dig_range[-1]])
        
        Gvals = G_dic()
        try:
            self.Bvals = {}
            for bkey in ARPES_dict['Brads']:
                self.Bvals[bkey] = lambda_gen(ARPES_dict['Brads'][bkey])
        except KeyError:
            self.Bvals = self.Bdic(dig_range,tmp_basis)

#        GB = self.matel_matrix(Bvals,Gvals,tmp_basis)
        
            
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
        
        nstates = len(self.TB.basis)
        phi = self.ph[int(cube[0]/nstates)]
        th = self.th[i]
        Mtmp = np.zeros((2,3),dtype=complex)
        
        psi = self.Ev[int(cube[0]/nstates),:,int(cube[0]%nstates)]
        
        for coeff in list(enumerate(psi)):

            if abs(coeff[1])>tol:
                o = basis[coeff[0]]

                L = [lp for lp in ([o.l-1,o.l+1] if (o.l-1)>=0 else [o.l+1])]
                pref = o.sigma*coeff[1]*np.exp((-self.mfp+0.0j*self.kz)*abs(o.depth))#*np.exp(1.0j*(-self.kz*o.pos[2]-self.X[int(self.pks[i,0]),int(self.pks[i,1])]*o.pos[0]-self.Y[int(self.pks[i,0]),int(self.pks[i,1])]*o.pos[1]))
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
        
        self.update_resolution(ARPES_dict['resolution'])
        
        w = np.linspace(*self.cube[2])
        pol = pol_2_sph(ARPES_dict['pol'])
        
        if ARPES_dict['spin']!=None:
            sv = ARPES_dict['spin'][1]/np.linalg.norm(ARPES_dict['spin'][1])
            th = np.arccos(sv[2])
            ph = np.arctan2(sv[1],sv[0])
            Smat = np.array([[np.cos(th/2),np.exp(-1.0j*ph)*np.sin(th/2)],[np.sin(th/2),-np.exp(-1.0j*ph)*np.cos(th/2)]])
            Mspin = np.swapaxes(np.dot(Smat,self.Mk),0,1)
 
        SE = abs(poly(self.pks[:,2],ARPES_dict['SE'])) #absolute value mandates that self energy be particle-hole symmetric, i.e. SE*(k,w) = -SE(k,-w). Here we define the imaginary part explicitly only!
        if self.T[0]:
            fermi = vf(w/(kb*self.T[1]/q))
        else:
            fermi = np.ones(self.cube[2][2])
        I = np.zeros((self.cube[1][2],self.cube[0][2],self.cube[2][2]))
        if ARPES_dict['spin'] is None:
            for p in range(len(self.pks)):
                if abs(self.Mk[p]).max()>0:
                    I[int(np.real(self.pks[p,0])),int(np.real(self.pks[p,1])),:]+= (abs(np.dot(self.Mk[p,0,:],pol))**2 + abs(np.dot(self.Mk[p,1,:],pol))**2)*np.imag(-1./(np.pi*(w-self.pks[p,2]+1.0j*(SE[p]+0.0005))))*fermi
        else:
            for p in range(len(self.pks)):
                if abs(Mspin[p]).max()>0:
                    I[int(np.real(self.pks[p,0])),int(np.real(self.pks[p,1])),:]+= abs(np.dot(Mspin[p,int((ARPES_dict['spin'][0]+1)/2),:],pol))**2*np.imag(-1./(np.pi*(w-self.pks[p,2]-SE[p]+0.01j)))*fermi
        kxg = self.dk/(self.cube[0][1]-self.cube[0][0])
        kyg = self.dk/(self.cube[1][1]-self.cube[1][0])
        wg = self.dE/(self.cube[2][1]-self.cube[2][0])
        
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
                tmp = self.TB.basis[o].copy()
                proj_arr = np.zeros(np.shape(tmp.proj),dtype=float)
                for p in range(len(tmp.proj)):
                    pnew = (tmp.proj[p][0]+1.0j*tmp.proj[p][1])*np.exp(-1.0j*tmp.proj[p][-1]*self.ang)
                    tmp_proj = np.array([np.around(np.real(pnew),5),np.around(np.imag(pnew),5),tmp.proj[p][2],tmp.proj[p][3]])
                    proj_arr[p] = tmp_proj
                tmp_base.append(tmp)
                tmp_base[-1].proj = proj_arr
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
        '''
        
        if spin_ax is not None:
            spin_mat = np.zeros((len(self.TB.basis),len(self.TB.basis)))
            ax_dict = {'x':[np.pi/2,0.0],'y':[np.pi/2,np.pi/2],'z':[0,0]}
            if abs(self.ang)>0:
                ax_dict[spin_ax][1]-=self.ang
            US = np.array([[-np.cos(ax_dict[spin_ax][0]/2)*np.exp(-1.0j*ax_dict[spin_ax][1]),np.sin(ax_dict[spin_ax][0]/2)],[np.sin(ax_dict[spin_ax][0]/2)*np.exp(-1.0j*ax_dict[spin_ax][1]),np.cos(ax_dict[spin_ax][0]/2)]])
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
    if len(args)==0:
        return 0
    else:
        return x**(len(args)-1)*args[-1] + poly(x,args[:-1])
    
    
    
    
    
    
    
    
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