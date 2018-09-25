#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


Created on Tue Feb 27 08:01:41 2018

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


import sys

sys.path.append('C:/Users/rday/Documents/TB_ARPES/2018/TB_ARPES_2018/TB_ARPES-master/')



import numpy as np
import ubc_tbarpes.build_lib as build_lib
import ubc_tbarpes.ARPES_lib as ARPES

'''
Very limited model of graphene, only considering Carbon-2pz orbitals. Rather than the normal
lattice, we are considering a supercell with the root(3) R30 folding, which has 6 rather than 
2 Carbon atoms.
The basis is designed as shown here
        5 o      
         /
     4  o
        |      
      3 o
       / \
   1  o 2 o
      |
   0  o




'''


def rndom_offset(matels,LB,DE):
    '''
    Quick function for introducing a random-energy offset to one of the supercell atoms.
    args:
        matels: tight-binding matrix elements as the mat_els attribute of a TB object
        LB: int (length of orbital basis)
        DE: float (energy offset of the randomly chosen site)
    '''
    index = int(np.random.random()*LB)
    print('Energy offset of {:0.04f} eV added to orbital {:d}'.format(DE,index))
    tmp_matels = matels
    for hi in range(len(matels)):
        if matels[hi].i==index and matels[hi].i==matels[hi].j:
            for ii in range(len(matels[hi].H)):
                if np.linalg.norm(matels[hi].H[ii][:3])==0.0:
                    tmp_matels[hi].H[ii] = [0.0,0.0,0.0,matels[hi].H[ii][-1]+DE]
    return tmp_matels
    


if __name__=="__main__":
    
    spin_args = {'bool':False,'soc':True,'lam':{0:0}}

 ######################### LATTICE DEFINITION ###############################   
    a,c =  2.46,10.0
   # avec = np.array([[-a/2,a*np.sqrt(3/4.),0.0],[a/2,a*np.sqrt(3/4.),0.0],[0.0,0.0,c]]) #standard unit cell
    avec = np.array([[-a*1.5,a*np.sqrt(0.75),0],[0,np.sqrt(3)*a,0],[0,0,c]]) #supercell

#    Basis_args = {'atoms':[0,0],
#			'Z':{0:6},
#			'orbs':[["21z"],["21z"]],
#			'pos':[np.zeros(3),np.array([0.0,a/np.sqrt(3.0),0.0])],
#            'spin':spin_args}
    
    ###SUPERCELL
    Basis_args = {'atoms':[0,0,0,0,0,0],
			'Z':{0:6},
			'orbs':[["21z"],["21z"],["21z"],["21z"],["21z"],["21z"]],
			'pos':[np.zeros(3),np.array([0.0,a/np.sqrt(3.0),0.0]),np.array([a*0.5,a*np.sqrt(0.75),0]),np.array([0.5*a,2.5/np.sqrt(3)*a,0]),np.array([a,np.sqrt(3)*a,0]),np.array([a,np.sqrt(1./3)*a,0])],
            'spin':spin_args}

 ######################### LATTICE DEFINITION ###############################   
 
 ######################### K-PATH DEFINITION ###############################   
    
    K = np.array([0.85138012,1.47463363,0])
    M = np.array([0,1.47463363,0])
    K_args = {'type':'A',
          'avec':avec,
			'pts':[K,G,M,K],
			'grain':81,
			'labels':['K','$\Gamma$','M','K']}
 ######################### K-PATH DEFINITION ###############################   

    
######################### HAMILTONIAN DEFINITION ##############################   
    SK = {"021":-0.44,"002211S":0.0,"002211P":-3.07} 
    CUT,REN,OFF,TOL=1.9,1,0.0,0.001	
    
    Ham_args = {'type':'SK',
			'V':SK,
          'avec':avec,
			'cutoff':CUT,
			'renorm':REN,
			'offset':OFF,
			'tol':TOL,
			'spin':spin_args}
    
######################### HAMILTONIAN DEFINITION ##############################   


######################### ARPES EXP. DEFINITION ##############################   

    
    Kpt =0.0# 1.7027
    ARPES_args = {'cube':{'X':[Kpt-0.2,Kpt+0.2,101],'Y':[-0.2,0.2,101],'kz':0.0,'E':[-1,0.5,200]},
            'SE':[0.02,0.00],
            'directory':'C:\\Users\\rday\\Documents\\TB_ARPES\\2018\\TB_ARPES_2018\\FeSe',
            'hv': 50,
            'pol':np.array([0,np.sqrt(0.5),-np.sqrt(0.5)]),
            'mfp':7.0,
            'resolution':{'E':0.02,'k':0.003},
            'T':[False,10.0],
            'W':4.0,
            'angle':0.0,
            'spin':None,
            'slice':[False,-0.2]}
    
    
######################### ARPES EXP. DEFINITION ##############################   


############################### BUILD MODEL #################################   

         
    Basis_args = build_lib.gen_basis(Basis_args)
    Kobj = build_lib.gen_K(K_args)
    TB = build_lib.gen_TB(Basis_args,Ham_args,Kobj)
    
    TB.mat_els = rndom_offset(TB.mat_els,len(TB.basis),0.5) #ADD RANDOM-ON SITE ENERGY of 500 meV
    
    TB.solve_H() #solve banfstructure over K-path defined by the K_args dictionary
    TB.plotting() #plot the bandstructure
    
############################### BUILD MODEL #################################   

############################ RUN ARPES EXPERIMENT #############################   


    ARPES_expmt = ARPES.experiment(TB,ARPES_args)
    ARPES_expmt.plot_gui(ARPES_args)
