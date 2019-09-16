# -*- coding: utf-8 -*-

#Created on Fri Apr 27 12:37:56 2018
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
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.



import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.tri as mtri

import chinook.Ylm as Ylm






class wavefunction:
    '''
    This class acts to reorganize basis and wavefunction information in a more
    suitable data structure than the native orbital class, or the sake of plotting
    orbital wavefunctions. The relevant eigenvector can be redefined, so long as it
    represents a projection onto the same orbital basis set as defined previously.
    
    *args*:

        - **basis**: list of orbital objects
        
        - **vector**: numpy array of complex float, eigenvector projected onto the basis orbitals
   
    '''
    
    def __init__(self,basis,vector):
        if len(basis)==len(vector):
            self.basis = basis
            self.centres,self.centre_pointer = self.find_centres()
            self.harmonics,self.harmonic_pointer,self.projections =self.find_harmonics()
            
            self.vector = vector
        else:
            print('ERROR: incompatible basis and vector input. Check that both have same length.')
        
    def redefine_vector(self,vector):

        '''
        Update vector definition

        *args*:

            - **vector**: numpy array of complex float, same length as self.vector

        ***
        '''

        try:
            self.vector[:] = vector
        except ValueError:
            print('Error: Input vector is not of the same shape as original selection. Please check input vector.')
            
        
    
    def find_harmonics(self):
        '''
        Create a pointer array of basis indices and the associated spherical harmonics, as well as
        aa more convenient vector form of the projections themselves, as lists of complex float
        
        *return*:

            - **all_lm**: list of int, l,m pairs of all spherical harmonics relevant to calculation
            
            - **lm_pointers**: list of int, pointer indices relating each basis orbital projection to the 
            lm pairs in *all_lm*
            
            - **projectors**: list of arrays of complex float, providing the complex projection of basis
            onto the related spherical harmonics
        
        ***
        '''
        all_lm = []
        lm_pointers = []
        projectors = []
        for o in self.basis:
            proj_pointers = np.zeros(len(o.proj))
            proj_vals = np.zeros(len(o.proj),dtype=complex)
            for oi in range(len(o.proj)):
                proj_vals[oi] = o.proj[oi][0]+1.0j*o.proj[oi][1]
                lm = np.array([o.proj[oi][2],o.proj[oi][3]]).astype(int)
                try:
                    d_lm = np.linalg.norm(np.array([lm_ii - lm for lm_ii in all_lm]),axis=1)
                    if d_lm.min()==0:
                        index = np.where(d_lm==0)[0][0]
                        proj_pointers[oi]=index
                    else:
                        all_lm.append(lm)
                        proj_pointers[oi] = len(all_lm)-1
                except ValueError:
                    all_lm.append(lm)
                    proj_pointers[0] = 0
            lm_pointers.append(list(proj_pointers.astype(int)))
            projectors.append(proj_vals)
        return all_lm,lm_pointers,projectors
        
            
    def find_centres(self):
        '''
        Create a Pointer array of basis indices and the centres of these basis orbitals.
        
        *return*:

            - **all_centres**: list of numpy array of length 3, indicating unique positions in the basis set
            
            - **centre_pointers**: list of int, indicating the indices of position array, associated with the
            location of the related orbital in real space.
        
        '''
        all_centres = []
        centre_pointers = []
        for o in self.basis:
            centre = o.pos
            try:
                d_centres = np.linalg.norm(np.array([centre-ac for ac in all_centres]),axis=1)
                if d_centres.min()==0.0:
                    index = np.where(d_centres==0)[0][0]
                    centre_pointers.append(index)
                else:
                    all_centres.append(centre)
                    centre_pointers.append(len(all_centres)-1)
            except ValueError:
                all_centres.append(centre)
                centre_pointers.append(0)
            
        return all_centres,centre_pointers
    
    
    def calc_Ylm(self,th,ph):

        '''
        Calculate all spherical harmonics needed for present calculation
        
        *return*:

            - numpy array of complex float, of shape (len(self.harmonics),len(th))
        
        ***
        '''
        return np.array([Ylm.Y(int(lm[0]),int(lm[1]),th,ph) for lm in self.harmonics])
        
        
                
            
    def triangulate_wavefunction(self,n,plotting=True,ax=None):

        '''
        Plot the wavefunction stored in the class attributes as self.vector as a projection
        over the basis of spherical harmonics. The radial wavefunctions are not explicitly included,
        in the event of multiple basis atom sites, the length scale is set by the mean interatomic 
        distance. The wavefunction phase is encoded in the colourscale of the mesh plot. The user
        sets the smoothness of the orbital projection by the integer argument *n*
        
        *args*:

            - **n**: int, number of angles in the mesh: Theta from 0 to pi is divided 2n times, and
            Phi from 0 to 2pi is divided 4n times
            
        *kwargs*:

            - **plotting**: boolean, turn on/off to display plot
            
            - **ax**: matplotlib Axes, for plotting on existing plot
            
            
        *return*:

            - **vertices**: numpy array of float, shape (len(centres), len(th)*len(ph), 3) locations of vertices
            
            - **triangulations**: numpy array of int, indicating the vertices connecting each surface patch
            
            - **colours**: numpy array of float, of shape (len(centres),len(triangles)) encoding the orbital phase for each surface patch of the plotting
        
            - **ax**: matplotlib Axes, for further modifications
        
        ***
        '''
        th,ph = make_angle_mesh(n)
        all_Ylm = self.calc_Ylm(th,ph)
        
        if len(self.centres)>1:
            ad = 0.5*np.mean(np.array([np.linalg.norm(self.centres[i]-self.centres[j]) for i in range(len(self.centres)) for j in range(i,len(self.centres))]))
        else:
            ad = 4.0
            
        ncentres = len(self.centres)
        vertices = np.zeros((ncentres,len(th),3))
        radii = np.zeros((ncentres,len(th)),dtype=complex)
        triangulations = mtri.Triangulation(th,ph)
        colours = []
        
        for bi in range(len(self.basis)):

            radii[self.centre_pointer[bi],:] += np.sum(np.array([all_Ylm[self.harmonic_pointer[bi][j]]*self.vector[bi]*self.projections[bi][j] for j in range(len(self.harmonic_pointer[bi]))]),axis=0)
        
        rescale = ad/np.mean(abs(radii)**2)
        for ni in range(ncentres):
            vertices[ni,:,:]+=rescale*np.array([abs(radii[ni])**2*np.cos(ph)*np.sin(th),abs(radii[ni])**2*np.sin(th)*np.sin(ph),abs(radii[ni])**2*np.cos(th)]).T
            colours.append(col_phase(radii[ni,triangulations.triangles][:,1]))
            vertices[ni,:]+=self.centres[ni]
            
        colours = np.array(colours)
        if plotting:
            
            _,ax = self.plot_wavefunction(vertices,triangulations,colours,plot_ax=ax)
            
        return vertices,triangulations,colours,ax
    
    
    def plot_wavefunction(self,vertices,triangulations,colours,plot_ax = None,cbar_ax= None):
        '''
        Plotting function, for visualizing orbitals.
        
        *args*:

            - **vertices**: numpy array of float, shape (len(centres), len(th)*len(ph), 3) locations of vertices
            
            - **triangulations**: numpy array of int, indicating the vertices connecting each surface patch
            
            - **colours**: numpy array of float, of shape (len(centres),len(triangles)) encoding the orbital phase for each surface patch of the plotting
        
            - **plot_ax**: matplotlib Axes, for plotting on existing axes
            
            - **cbar_ax**: matplotlib Axes, for use in drawing colourbar
            
        *return*:

            - **plots**: list of plotted surfaces
            
            - **plot_ax**: matplotlib Axes, for further modifications
            
        ***
        '''
        ncentres = len(self.centres)
        plots = []
        if plot_ax is None:
            fig = plt.figure()
            plot_ax = fig.add_subplot(111,projection='3d')

        for ni in range(ncentres):
            plots.append(plot_ax.plot_trisurf(vertices[ni,:,0],vertices[ni,:,1],vertices[ni,:,2],triangles=triangulations.triangles,cmap=cm.hsv,antialiased=True,edgecolors='w',linewidth=0.2))
            plots[-1].set_array(colours[ni])
            plots[-1].set_clim(-np.pi,np.pi)
            
        plot_ax.set_xlabel('X')
        plot_ax.set_ylabel('Y')
        plot_ax.set_zlabel('Z')
        plt.colorbar(plots[-1],ax=plot_ax,cax=cbar_ax)
        
        return plots,plot_ax
            

def make_angle_mesh(n):
    '''
    Quick utility function for generating an angular mesh over spherical surface
    
    *args*:

        - **n**: int, number of divisions of the angular space
        
    *return*:

        - **th**: numpy array of 2n float from 0 to pi
        
        - **ph**: numpy array of 4n float from 0 to 2pi
    
    ***
    '''
    th = np.linspace(0,np.pi,2*n)
    ph = np.linspace(0,2*np.pi,4*n)
    th,ph = np.meshgrid(th,ph)
    th,ph = th.flatten(),ph.flatten()
    return th,ph


def col_phase(vals):

    '''
    Define the phase of a complex number
    
    *args*:

        - **vals**: complex float, or numpy array of complex float
        
    *return*:

        - float, or numpy array of float of same shape as vals, from -pi to pi
    
    ***
    '''
    x,y=np.real(vals),np.imag(vals)
    return np.arctan2(y,x)


def rephase_wavefunctions(vecs,index=-1):

    '''
    The wavefunction at different k-points can choose an arbitrary phase, as can 
    a subspace of degenerate eigenstates. As such, it is often advisable to choose
    a global phase definition when comparing several different vectors. The user here
    passes a set of vectors, and they are rephased. The user has the option of specifying
    which basis index they would like to set the phasing. It is essential however that the
    projection onto at least one basis element is non-zero over the entire set  of vectors 
    for this rephasing to work.
    
    *args*:

        - **vecs**: numpy array of complex float, ordered as rows:vector index, columns: basis index
        
    *kwargs*:

        - **index**: int, optional choice of basis phase selection
        
    *return*:

        - **rephase**: numpy array of complex float of same shape as *vecs*
    
    ***
    '''
    
    rephase = np.copy(vecs)
    if index>-1:
        #check that user has selected a viable phase choice
        if abs(vecs[:,index]).min()<1e-10:
            print('Warning, the chosen basis index is invalid. Please make another selection.\n')
            print('Finite projection onto the basis element of choice must be finite. If you are\n')
            print('unsure, the computer can attempt to make a viable selection in the absence of\n')
            print('an indicated basis index.')
            return rephase
    
    else:
        min_projs = np.array([abs(vecs[:,i]).min() for i in range(np.shape(vecs)[0])])
        index = np.where(min_projs>0)[0][0]
    phase_factors = np.conj(vecs[:,index])/abs(vecs[:,index])
    rephase = np.einsum('ij,i->ij',rephase,phase_factors)
    
    return rephase
    
            
        
    