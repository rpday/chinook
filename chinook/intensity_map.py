#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Created on Mon Mar 18 13:27:50 2019


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
import datetime as dt
try:
    import h5py
except ImportError:
    print('Warning: no h5py found. Visit https://www.docs.h5py.org/en/stable/build.html for download instructions.\nProceed using text-file output only.')
    

class intensity_map:
    
    def __init__(self,index,Imat,ROI,T,hv,pol,dE,dk,self_energy=None,spin=None,rot=0.0,notes=None):
        self.index = index
        self.ROI = ROI
        self.T = T
        self.hv = hv
        self.pol = pol
        self.spin = spin
        self.Imat = Imat
        self.rot = rot
        self.self_energy = self_energy
        self.dE = dE
        self.dk = dk
        if type(notes)==str:
            self.notes = notes
        else:
            self.notes = 'N/A'
            
            
    def save_hd5f(self,directory):
        '''
        Save the dataset to an hdf5 file. 
        
        '''
        
        h5name = directory+'chinook_intensity.hdf5'
        newfile = h5py.File(h5name,'a')
        mydata = newfile.create_dataset('intensity/intensity_map',shape=np.shape(np.squeeze(self.Imat)),dtype='f',data=np.squeeze(self.Imat))
        mydata = self.build_h5meta(mydata)
        newfile.close()
        
    def build_h5meta(self,mydata):
        mydata.attr['Temperature (K)'] = self.T
        mydata.attr['Photon Energy (eV)'] = self.hv
        mydata.attr['Polarization (a.u.)'] = self.pol
        mydata.attr['Spin Projection (a.u.)'] = self.spin
        mydata.attr['Sample Azimuthal Rotation (deg)'] = self.rot*180/np.pi
        mydata.attr['Self Energy Parameters'] = self.self_energy
        mydata.attr['Energy Resolution (eV)'] = self.dE
        mydata.attr['Momentum Resolution (1/A)'] = self.dk
        mydata.attr['Notes'] = self.notes
        mydata.attr['Momentum X (1/A)'] = self.ROI[0]
        mydata.attr['Momentum Y (1/A)'] = self.ROI[1]
        mydata.attr['Energy (eV)'] = self.ROI[2]
        return mydata
        
        
        
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
        dim_min = min([a for a in np.shape(self.Imat)])
        if len(np.shape(self.Imat))==3 and dim_min>1:
            for i in range(np.shape(self.Imat)[2]):
                filename = directory + '_{:d}.txt'.format(i)
                self.write_2D_Imat(filename,i)
        else:
            self.write_2D_Imat(directory+'_0.txt',-1)
    
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
            imat = np.squeeze(self.Imat)
                
        with open(filename,"w") as destination:        
            
            for ii in range(np.shape(imat)[0]):
                tmpline = " ".join(['{:0.04f}'.format(vi) for vi in imat[ii,:]])
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
            tofile.write('Polarization: [{:s}, {:s}, {:s}]\n'.format(*self.pol.astype(str)))
            tofile.write('Energy Resolution: {:0.04f} eV\n'.format(self.dE))
            tofile.write('Momentum Resolution: {:0.04f} 1/A\n\n'.format(self.dk))
            tofile.write('Kx Domain: {:0.04f} -> {:0.04f} 1/A, N = {:d}\n'.format(*self.ROI[0]))
            tofile.write('Ky Domain: {:0.04f} -> {:0.04f} 1/A, N = {:d}\n'.format(*self.ROI[1]))
            tofile.write('Kz: {:0.04f} 1/A\n'.format(self.ROI[3][0]))
            tofile.write('Energy Domain: {:0.04f} -> {:0.04f} eV, N = {:d}\n\n'.format(*self.ROI[2]))
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
                    w = np.linspace(*self.ROI[2])
                    SE = self.self_energy[1](w)
                else:
                    w = self.self_energy[1]
                    SE = self.self_energy[2]
                tofile.write('Energy (eV) | Self Energy (eV) \n')
                for ii in range(w):
                    tofile.write('{:s}\t {:s}\n'.format(str(w[ii]),str(SE[ii])))
            
        tofile.close()
        
        
    def copy(self):
        '''
        Copy-by-value of the intensity map object. 
        *return*:
            - *intensity_map* object with identical attributes to self.
        '''
        return intensity_map(self.index,self.Imat,self.ROI,self.T,self.hv,self.pol,self.dE,self.dk,self.self_energy,self.spin,self.rot)
