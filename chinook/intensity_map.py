#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 13:27:50 2019

@author: ryanday
"""
import numpy as np
import datetime as dt

class intensity_map:
    
    def __init__(self,index,Imat,cube,kz,T,hv,pol,dE,dk,self_energy=None,spin=None,rot=0.0,notes=None):
        self.index = index
        self.cube = cube
        self.kz = kz
        self.T = T
        self.hv = hv
        self.pol = pol
        self.spin = spin
        self.Imat = Imat
        self.rot = rot
        self.self_energy = self_energy
        self.dE = dE
        self.dk = dk
        self.notes = notes
        
    def save_map(self,directory):
        '''
        Save the intensity map: if 2D, just a single file, if 3D, each constant-energy
        slice is saved separately.

        *args*:
            - **directory**: string, directory for saving intensity map to
        
        *return*:
            - boolean
        '''
        self.write_meta(directory+'_meta.txt')
        if len(np.shape(self.Imat))==3:
            for i in range(np.shape(self.Imat)[2]):
                filename = directory + '_{:d}.txt'.format(i)
                self.write_2D_Imat(filename,i)
        else:
            self.write_2D_Imat(self,directory+'_0.txt',-1)
    
        return True
    
    def write_2D_Imat(self,filename,index):
        '''
        Sub-function for producing the textfiles associated with a 2dimensional numpy array of float

        *args*:
            - **filename**: string, indicating destination of file
            
            - **index**: int, energy index of map to save, if -1, then just a 2D map, and save the whole
            thing
            
        *return*:
            - boolean
        
        '''
        if index>-1:
            imat = self.Imat[:,:,index]
        else:
            imat = self.Imat
                
        with open(filename,"w") as destination:        
            
            for ii in range(np.shape(imat)[0]):
                tmpline = " ".join(map(str,imat))
                tmpline+="\n"
                destination.write(tmpline)
        destination.close()
        return True
    
    def write_meta(self,destination):
        '''
        Write meta-data file for ARPES intensity calculation.
        
        *args*:
            - **destination**: string, file-lead
            
        *return*:
            - boolean
        '''
        
        with open(destination,'w') as tofile:
            now = dt.datetime.now()
            now_time = now.strftime('%H:%M:%S %d/%m/%y')
            tofile.write('ARPES calculation: '+now_time+'\n')
            tofile.write('Calculation notes: {:s}\n\n'.format(self.notes))
            tofile.write('Temperature: {:0.02f} K\n'.format(self.T))
            tofile.write('Photon Energy: {:0.04f} eV\n'.format(self.hv))
            tofile.write('Polarization: [{:s}, {:s}, {:s}]\n'.format(self.pol.astype(str)))
            tofile.write('Energy Resolution: {:0.04f} eV\n'.format(self.dE))
            tofile.write('Momentum Resolution: {:0.04f} 1/A\n\n'.format(self.dk))
            tofile.write('Kx Domain: {:0.04f}, {:0.04f}, {:0.04f} 1/A\n'.format(*self.cube[0]))
            tofile.write('Ky Domain: {:0.04f}, {:0.04f}, {:0.04f} 1/A\n'.format(*self.cube[1]))
            tofile.write('Kz: {:0.04f} 1/A\n'.format(self.kz))
            tofile.write('Energy Domain: {:0.04f}, {:0.04f}, {:0.04f} eV\n\n'.format(*self.cube[2]))
            tofile.write('Sample Rotation: {:0.04f}\n'.format(self.rot))
            if self.spin is None:
                tofile.write('Spin Projection: None\n')
                tofile.write('Spin Axis: None \n')
            else:
                tofile.write('Spin Projection: {:d}\n'.format(self.spin[0]))
                tofile.write('Spin Axis: {:0.04f}, {:0.04f}, {:0.04f}\n'.format(*self.spin[1:]))
            
            if self.self_energy[0]=='poly':
                tofile.write('Self Energy: '+'+'.join(['{:0.04f}w^{:d}'.format(self.self_energy[i],i-1) for i in range(1,len(self.self_energy))])+'\n')
            elif self.self_energy[0] == 'constant':
                tofile.write('Self Energy: {:0.04f}\n'.format(self.self_energy[1]))
            else:
                if self.self_energy[0] == 'func':
                    w = np.linspace(*self.cube[2])
                    SE = self.self_energy[1](w)
                else:
                    w = self.self_energy[1]
                    SE = self.self_energy[2]
                tofile.write('Energy (eV) | Self Energy (eV) \n')
                for ii in range(w):
                    tofile.write('{:s}\t {:s}\n'.format(str(w[ii]),str(SE[ii])))
            
        tofile.close()
        
        
    def copy(self):
        return intensity_map(self.index,self.Imat,self.cube,self.kz,self.T,self.hv,self.pol,self.dE,self.dk,self.self_energy,self.spin,self.rot)
