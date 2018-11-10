#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 15:41:01 2017

@author: ryanday
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
import ubc_tbarpes.H_library as Hlib
from operator import itemgetter
import datetime as dt
from itertools import compress


'''
Tight-Binding Utility module

'''


class H_me:
    
    def __init__(self,i,j):
        self.i = i
        self.j = j
        self.H = []
    
    def append_H(self,R0,R1,R2,H):
        self.H.append([R0,R1,R2,H])
        
        
    def H2Hk(self): #this employs lambda functions to allow for rapid initialization of the Hamiltonian. 

        return lambda x: sum([complex(m[-1])*np.exp(1.0j*np.dot(x,np.array([m[0],m[1],m[2]]))) for m in self.H])
        
    def clean_H(self):
        tmp = self.H
        bools = [True]*len(tmp)
        for hi in range(len(tmp)-1):
            for hj in range(hi+1,len(tmp)):
                
                try:
                    norm = np.linalg.norm(np.array(tmp[hi])-np.array(tmp[hj]))
                    if abs(norm)<1e-5:
                        bools[hj] = False
                except IndexError:
                    continue
        return list(compress(tmp,bools))
    
    def copy(self):
        tmp_H = H_me(self.i,self.j)
        tmp_H.H = self.H[:]
    
    
        
class TB_model:
    
    
    def __init__(self,basis,H_args = None,Kobj = None):
        '''
        basis -- list of orbital objects
        H_args: dictionary for Hamiltonian build:
            {'type': "SK" ,"list", or "txt"
             if SK:
                 'V': dictionary of TB,
                 'avec': lattice vectors for generating neighbours
                 'cutoff':cutoff distance
                 'renorm': renormalization factor
                 'offset':offset
                 'tol': min value
             if 'list':
                 'list': list of Hamiltonian elements: [i,j,R1,R2,R3,Hij]
                 'cutoff': cutoff distance
                 'renorm': renormalization factor
                 'offset': offset
                 'tol':min value
                 
             if 'txt':
                 'filename':path to text file
                 'cutoff':cutoff distance
                 'renorm': renormalization factor
                 'offset':offset
                 'tol': min value
            
            'so':boolean True/False for whether spin orbit is to be included
        
        '''
        self.basis = basis #is a list of orbital objects
        if H_args is not None:
            self.avec = H_args['avec']

        self.mat_els = self.build_ham(H_args)
        self.Kobj = Kobj
        
    def copy(self):
        TB_copy = TB_model(self.basis,None,self.Kobj)
        TB_copy.avec = self.avec.copy()
        TB_copy.mat_els = self.mat_els[:]
        
        return TB_copy
        
            
    
    def build_ham(self,H_args):
        '''
        Buld the Hamiltonian using functions from H_library
        args:
            Htype: string either "SK" "list" or "txt" for Slater-Koster, python list or textfile
            args: Hamiltonian args-->for SK this will be library of Vxyz and list of cutoffs
                                  --> for list, the Hamiltonian should be passed, formatted as a list ready to go
                                  -->for txt it will be a filename and cutoff criterias
            spin: dictionary for including spin-degrees of freedom:
                bool: generate spin-duplicates
                so: boolean for spin-orbit coupling
                order: string 'NA', 'F' or 'A' for none, Ferromagnetic or antiferromagnetic respectively
        return:
            sorted list of matrix element objects.
            These objects have i,j attributes referencing the orbital basis indices, and a list
            of form [R0,R1,R2,Re(H)+1.0jIm(H)]
        '''
        if type(H_args)==dict:
            try:
                htmp = []
                if H_args['type'] == "SK":
                    htmp = Hlib.sk_build_2(H_args['avec'],self.basis,H_args['V'],H_args['cutoff'],H_args['tol'],H_args['renorm'],H_args['offset'])
#                    htmp = Hlib.sk_build(H_args['avec'],self.basis,H_args['V'],H_args['cutoff'],H_args['tol'],H_args['renorm'],H_args['offset'],H_args['spin']['bool'])
                elif H_args['type'] == "txt":
                    htmp = Hlib.txt_build(H_args['filename'],H_args['cutoff'],H_args['renorm'],H_args['offset'],H_args['tol'])
                elif H_args['type'] == "list":
                    htmp = H_args['list']
                if H_args['spin']['bool']:
                    h2 = Hlib.spin_double(htmp,len(self.basis))
                    htmp = htmp + h2   
                    if H_args['spin']['soc']:
                        so = Hlib.SO(self.basis)
                        htmp = htmp + so
                    if 'order' in H_args['spin']:
                        if H_args['spin']['order']=='F':
                            h_FM = Hlib.FM_order(self.basis,H_args['spin']['dS'])
                            htmp = htmp + h_FM
                        elif H_args['spin']['order']=='A':
                            h_AF = Hlib.AFM_order(self.basis,H_args['spin']['dS'],H_args['spin']['p_up'],H_args['spin']['p_dn'])
                            htmp = htmp + h_AF
                H = gen_H_obj(htmp)
                return H 
            except KeyError:
                print('Invalid dictionary input for Hamiltonian generation')
                return None
        else:
            return None
            

        
        
    def solve_H(self):
        '''
        This function diagonalizes the Hamiltonian over an array of kpoints. It uses the mat_el
        objects to quickly define lambda functions of k, which are then filled into the array
        and then diagonalized.
        
        returns: two arrays of eigenvalues and eigenvectors from the diagonalized Hamiltonian
        over the k array
        '''
        if self.Kobj is not None:
            Hmat = np.zeros((len(self.Kobj.kpts),len(self.basis),len(self.basis)),dtype=complex) #initialize the Hamiltonian
            
            for me in self.mat_els:
                Hfunc = me.H2Hk() #transform the array above into a function of k
                Hmat[:,me.i,me.j] = Hfunc(self.Kobj.kpts) #populate the Hij for all k points defined
        
            self.Eband,self.Evec = np.linalg.eigh(Hmat,UPLO='U') #diagonalize--my H_raw definition uses i<=j, so we want to use the upper triangle in diagonalizing
            return self.Eband,self.Evec
        else:
            print('You have not defined a set of kpoints over which to diagonalize.')
            return False
            
        
        
    def plotting(self,win_min=None,win_max=None,svlabel=None,title=None,lw=1.5,text=None): #plots the band structure. Takes in Latex-format labels for the symmetry points indicated in the main code
        fig=plt.figure()
        ax=fig.add_subplot(111)
        plt.axhline(y=0,color='k',lw=lw,ls='--')
        for b in self.Kobj.kcut_brk:
            plt.axvline(x = b,color = 'k',ls='--',lw=lw)
        for i in range(len(self.basis)):
            plt.plot(self.Kobj.kcut,np.transpose(self.Eband)[i,:],color='w',lw=lw)

        plt.xticks(self.Kobj.kcut_brk,self.Kobj.labels)
        if win_max==None or win_min==None:
            plt.axis([self.Kobj.kcut[0],self.Kobj.kcut[-1],np.amin(self.Eband)-1.0,np.amax(self.Eband)+1.0])
        elif win_max !=None and win_min !=None:
            plt.axis([self.Kobj.kcut[0],self.Kobj.kcut[-1],win_min,win_max]) #hard coded right now for Bi2Se3, should revise
        if text is not None:
            props = dict(boxstyle='round',facecolor='wheat',alpha=0.5)
            ax.text(0.05,0.2,text,transform=ax.transAxes,fontsize=14,verticalalignment='top',bbox=props)
        if title is not None:
            plt.suptitle(title)   
        plt.ylabel("Energy (eV)")
        if svlabel is not None:
            plt.savefig(svlabel)
            
            
            
def gen_H_obj(htmp):
        htmp = sorted(htmp,key=itemgetter(0,1,2,3,4))

        
        H = []
        
        Hnow = H_me(0,0)
        Rij = np.zeros(3)
        
        for h in htmp:
            if h[0]!=Hnow.i or h[1]!=Hnow.j:
                H.append(Hnow)
                Hnow = H_me(int(np.real(h[0])),int(np.real(h[1])))
            Rij = np.real(h[2:5])
            Hnow.append_H(*Rij,h[5])
        H.append(Hnow)
        return H 
            

            
            


        
        
            
    
    