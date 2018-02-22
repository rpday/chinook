#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 15:41:01 2017

@author: ryanday
"""

import numpy as np
import matplotlib.pyplot as plt
import H_library as Hlib
from operator import itemgetter
from matplotlib import rc 



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
        

class TB_model:
    
    
    def __init__(self,basis,H_args,Kobj = None):
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
        self.mat_els = self.build_ham(H_args)
        self.Kobj = Kobj
            
    
    def build_ham(self,H_args):
        '''
        Buld the Hamiltonian using functions from H_library
        args:
            Htype: string either "SK" "list" or "txt" for Slater-Koster, python list or textfile
            args: Hamiltonian args-->for SK this will be library of Vxyz and list of cutoffs
                                  --> for list, the Hamiltonian should be passed, formatted as a list ready to go
                                  -->for txt it will be a filename and cutoff criterias
            so: boolean for spin-orbit coupling
        
        return:
            sorted list of matrix element objects.
            These objects have i,j attributes referencing the orbital basis indices, and a list
            of form [R0,R1,R2,Re(H)+1.0jIm(H)]
        '''
        htmp = []
        if H_args['type'] == "SK":
            htmp = Hlib.sk_build(H_args['avec'],self.basis,H_args['V'],H_args['cutoff'],H_args['tol'],H_args['renorm'],H_args['offset'],H_args['so'])
        elif H_args['type'] == "txt":
            htmp = Hlib.txt_build(H_args['filename'],H_args['cutoff'],H_args['renorm'],H_args['offset'],H_args['tol'])
        elif H_args['type'] == "list":
            htmp = H_args['list']
        if H_args['so']:
            h2 = Hlib.spin_double(htmp,len(self.basis))
                

            so = Hlib.SO(self.basis)
            htmp = htmp + h2+ so
        
        htmp = sorted(htmp,key=itemgetter(0,1,2,3,4))

        
        H = []
        
        Hnow = H_me(0,0)
        
        for h in htmp:
            if h[0]!=Hnow.i or h[1]!=Hnow.j:
                H.append(Hnow)
                Hnow = H_me(h[0],h[1]) 
            Hnow.append_H(h[2],h[3],h[4],h[5])
        H.append(Hnow)
        return H  
            
     
        
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
        plt.axhline(y=0,color='grey',lw=lw,ls='--')
        rc('font',**{'family':'serif','serif':['Palatino'],'size':20})
        rc('text',usetex = False) 
        for b in self.Kobj.kcut_brk:
            plt.axvline(x = b,color = 'grey',ls='--',lw=lw)
        for i in range(len(self.basis)):
            plt.plot(self.Kobj.kcut,np.transpose(self.Eband)[i,:],color='navy',lw=lw)

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
            
            
            


        
        
            
    
    