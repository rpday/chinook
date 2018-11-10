#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 11:08:31 2018

@author: ryanday

This script builds a UBC_TBARPES model from a Wannier90 calculation. The requisite inputs
are:
    
    seedname_hr.dat -- W90 Output File: Hamiltonian matrix
    seedname.win -- W90 Input File
    seedname.inwf -- W90 Input File, Input Wavefunctions with projections in Spherical Harmonics


"""

import numpy as np


orb_dict = {'s':'0','p':'1','d':'2','f':'3'}

class orbital:
    
    def __init__(self,index,label,pos,proj):
        self.index = index
        label_splt = label.split(':')        
        self.atom = int(label_splt[0])-1
        self.label = orb_dict[label_splt[1][0]]+label_splt[1][1:]
        self.pos = pos
        self.proj = proj
        



class wannier:
                    
    def __init__(self,hr_file,win_file,inwf_file):
        
        self.avec,self.pos = self.read_winfile(win_file)
        self.projections = self.read_inwfile(inwf_file)
        
        self.basis = self.gen_basis()
        
        self.hamiltonian = self.read_hrfile(hr_file)
        
        
        
    def read_winfile(self,fnm):
        '''
        Load and extract lattice, basis information from the Wannier90 seedname.win file
        pass the extracted strings to tidy_up_lattice to get the lattice vectors and basis 
        positions in a useful format. Then send to __init__
        
        '''
        
        read= False
        out = []
        with open(fnm,'r') as infile:
            for line in infile:
                line_splt = line.split(' ')
                if line_splt[0]=='begin':
                    buffer = [line_splt[1][:-1]]
                    read= True
                    continue
                if line_splt[0]=='end':
                    read =False
                    if len(buffer)>1:
                        out.append(buffer)
                    
                if read:
                    if buffer[0]=='unit_cell_cart' or buffer[0]=='atoms_cart':
                        buffer.append(line[:-1])
        lattice,atoms=self.tidy_up_lattice(out)
        return lattice,atoms
    
    def read_inwfile(self,fnm):
        '''
        Load projections in from the Wannier90 projection-initialization file,
        seedname.inwf. Each atom is labelled by a line of form:
           int         # int:char -> int:str
        The information we want is the final int:str which correspond to atom number:orbital label
        Then on the intervening lines are the orbital projections in format:
            int int  int  float   float
        which correspond to atom#   l   ml   Re(proj)    Im(proj)
        
        These are used to generate and return a dictionary with keys of form "atom:label"
        and items are array of projections in the UBC_TBARPES format
        '''
        proj_dict = {}
        read_proj = False
        proj = []
        label = ''
        with open(fnm,'r') as infile:
            for line in infile:
                tmp_st = line.split()
                if '->' in tmp_st:
                    read_proj = True
                    if len(proj)>0:
                        proj_dict[label] = np.array(proj)
                        proj = []
                    label = tmp_st[-1]
                    continue
                if read_proj:
                    proj.append([float(tmp_st[3]),float(tmp_st[4]),float(tmp_st[1]),float(tmp_st[2])])
        if len(proj)>0:
            proj_dict[label] = np.array(proj)                      
        return proj_dict
    
    def read_hrfile(self,fnm):
        '''
        Read the Wannier Hamiltonian text file and convert to the standard format of Hamiltonian.
        '''
        Hlist = []
        lc = 0
        with open(fnm,'r') as infile:
            for line in infile:
                if lc>19:
                    Hstr = line.split()
#                    if float(Hstr[-1])>1e-4:
#                        print('WARNING!! NON-ZERO IMAGINARY COMPONENT')
#                        print(Hstr)
                    int_vector = np.array([int(Hstr[0]),int(Hstr[1]),int(Hstr[2])])
                    i1,i2 = int(Hstr[3])-1,int(Hstr[4])-1
                    cvec = self.hopping_path(int_vector,i1,i2)
                    Hlist.append([i1,i2,*cvec,float(Hstr[-2])])
                lc+=1
        
        
        return Hlist
    
    def hopping_path(self,vector,i1,i2):
        lattice_vector = np.dot(vector,self.avec)
        return lattice_vector+ self.basis[i2].pos - self.basis[i1].pos 
    
    
    
    
    def gen_basis(self):
        basis = []
        for oi in self.projections:
            pos = self.pos[int(oi[0])-1]
            basis.append(orbital(len(basis),oi,pos,self.projections[oi]))
        return basis

    def tidy_up_lattice(self,out):
        '''
        Convert the plain-text formatted lattice and basis positions into
        float-type numpy arrays which can be more effectively used
        args:
            out -- list of data relating to lattice vectors [0] and basis atom positions [1]
        return:
            vectors numpy array of float for 3x3
            atoms numpy array of basis atom positions
        '''
            
            
        
        units = out[0][1]
        if units.upper()=='BOHR':
            mult = 0.529177
        else:
            mult = 1.0
        
        vectors = []
        atoms = []
        
        for i in range(3):
            aline = out[0][2+i].split()
            vectors.append([float(aline[0]),float(aline[1]),float(aline[2])])
        vectors = mult*np.array(vectors)
        
        for i in range(len(out[1])-2):
            aline = out[1][2+i].split()
            atoms.append([float(aline[1]),float(aline[2]),float(aline[3])])
        
        atoms = mult*np.array(atoms)
        return vectors,atoms
        
        
            
def write_ham(ham,fnm):
    
    with open(fnm,'w') as tofile:
        for hi in ham:
            line = '{:d},{:d},{:0.04f},{:0.04f},{:0.04f},{:0.04f}\n'.format(*hi)
            tofile.write(line)
    tofile.close()
    
if __name__ == "__main__":
    seedname = 'C:/Users/rday/Documents/TB_ARPES/2018/TB_ARPES_2018/TB_ARPES-master/examples/Bi2Se3/Bi2Se3'

    hr = seedname+'_hr.dat'
    inwf = seedname + '.inwf'
    win = seedname + '.win'
    new_wann = wannier(hr,win,inwf)                
                
                    
                
                
                
        


