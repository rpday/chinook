# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 15:22:36 2018

@author: rday
"""

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
import ubc_tbarpes.operator_library as ops

'''
Very limited model of graphene, only considering Carbon-2pz orbitals. Rather than the normal
lattice, we are considering a supercell with the root(3) R30 folding, which has 6 rather than 
2 Carbon atoms.
\
'''

#np.array([0,a/np.sqrt(3)+da,0]),
#                  np.array([(a/np.sqrt(3)+da)*np.sqrt(0.75),(a/np.sqrt(3)+da)*0.5,0]),
#                  np.array([(a/np.sqrt(3)+da)*np.sqrt(0.75),a*2.5/np.sqrt(3)-0.5*da,0]),
#                  np.array([0,2*a/np.sqrt(3)-da,0]),
#                  np.array([-(a/np.sqrt(3)+da)*np.sqrt(0.75),a*2.5/np.sqrt(3)-0.5*da,0]),
#                  np.array([-(a/np.sqrt(3)+da)*np.sqrt(0.75),(a/np.sqrt(3)+da)*0.5,0]),
#                  np.array([0,0,1.75])]



if __name__=="__main__":
    
    spin_args = {'bool':False,'soc':True,'lam':{0:0}}

 ######################### LATTICE DEFINITION ###############################
    cc = 1.41   
    c = 20.006659
    avec = np.array([[3*np.sqrt(3)/2*cc,3*cc/2.,0],[-3*np.sqrt(3)*cc/2.,3*cc/2.,0],[0,0,c]])

    pos = np.array([[0.0,cc,0.0],
                    [0.0,-cc,0.0],
                    [cc*np.sqrt(3)/2.,cc/2.,0],
                    [cc*np.sqrt(3)/2.,-cc/2.,0],
                    [-cc*np.sqrt(3)/2.,cc/2.,0],
                    [-cc*np.sqrt(3)/2.,-cc/2.,0],
                    [0,0,1.7430]])
    
    G,K = np.array([0,0,0]),np.array([0,1,0])
    
    ###SUPERCELL
    da = 0.0#2.85e-3
    Basis_args = {'atoms':[0,0,0,0,0,0,1],
			'Z':{0:6,1:3},
			'orbs':[["21z"],["21z"],["21z"],["21z"],["21z"],["21z"],["20"]],
			'pos':[pi for pi in pos],
            'spin':spin_args}

 ######################### LATTICE DEFINITION ###############################   
 
 ######################### K-PATH DEFINITION ###############################   
    
    K_args = {'type':'F',
          'avec':avec,
			'pts':[K,G,K],
			'grain':100,
			'labels':['M','$\Gamma$','K']}
 ######################### K-PATH DEFINITION ###############################   

    
######################### HAMILTONIAN DEFINITION ##############################   
    t0=-2.6
    SK = {"021":-1,"002211S":0.1,"002211P":t0,"120":0,"012210S":0.0} 
    CUT,REN,OFF,TOL=1.5,1,1.6,0.001	
    
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

    
    Kpt =(0,0)#(0.8514,1.4746)
    dk = 0.2
    Nk = 101
    ARPES_args = {'cube':{'X':[Kpt[0]-dk,Kpt[0]+dk,Nk],'Y':[Kpt[1]-dk,Kpt[1]+dk,Nk],'kz':0.0,'E':[-1.5,0.5,200]},
            'SE':[0.02,0.00],
            'directory':'C:Users/rday/Documents/graphene/',
            'hv': 50,
            'pol':np.array([0,1,0]),
            'mfp':7.0,
            'resolution':{'E':0.02,'k':0.003},
            'T':[True,10.0],
            'W':4.0,
            'angle':0.0,
            'spin':None,
            'slice':[False,-0.2]}
    
    
######################### ARPES EXP. DEFINITION ##############################   


############################### BUILD MODEL #################################   

         
    Basis_args = build_lib.gen_basis(Basis_args)
    Kobj = build_lib.gen_K(K_args)
    TB = build_lib.gen_TB(Basis_args,Ham_args,Kobj)
    
#    TB.mat_els = rndm_offset(TB.mat_els,len(TB.basis),DE = -0.5) #ADD RANDOM-ON SITE ENERGY
#    TB.mat_els = rndm_bond(TB.mat_els,len(TB.basis),DE = -0.5) #ADD RANDOM-HOPPING ENERGY
#    TB.mat_Els = mass_term(TB.mat_els,DE=0.9)
#    TB.mat_els = kek_O(TB.mat_els,0.1) #CHANGE HOPPINGS FOR 1 RING
    
#    TB.mat_els =  rebond(TB.mat_els,a/np.sqrt(3),t0,0.5)
    TB.solve_H() #solve banfstructure over K-path defined by the K_args dictionary
#    print(TB.Eband[0,4]-TB.Eband[0,2])
#    TB.plotting() #plot the bandstructure
#    s =np.array([bi.index for bi in TB.basis if bi.label[-1]=='0' ])
#    Os=ops.fatbs(s,TB,Kobj=TB.Kobj,vlims=(0,1),Elims=(-1,0.5),degen=False)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(7,15):
        if i!=8 and i!=9:
            ax.plot(kspace,LiC6[i,:,1],c='k')
    for i in range(7):
        ax.plot(kspace,TB.Eband[:,i],c='r')
    
############################### BUILD MODEL #################################   

############################ RUN ARPES EXPERIMENT #############################   

#
#    ARPES_expmt = ARPES.experiment(TB,ARPES_args)
#    ARPES_expmt.plot_gui(ARPES_args)
