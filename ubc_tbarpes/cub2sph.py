#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 09:21:14 2017

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
'''
Function to project the eigenstate written in cubic harmonics onto the basis of spherical harmonics

'''

import numpy as np


#projections of different orbitals written in order of ml=-l,...,0,...,l
Y={"1x":np.array([np.sqrt(0.5),0,-np.sqrt(0.5)]),
   "1y":np.array([1.0j*np.sqrt(0.5),0,1.0j*np.sqrt(0.5)]),
   "1z":np.array([0,1,0]),
   "2xz":np.array([0,np.sqrt(0.5),0,-np.sqrt(0.5),0]),
   "2yz":np.array([0,np.sqrt(0.5)*1.0j,0,np.sqrt(0.5)*1.0j,0]),
   "2xy":np.array([np.sqrt(0.5)*1.0j,0,0,0,-np.sqrt(0.5)*1.0j]),
   "2ZR":np.array([0,0,1,0,0]),
   "2XY":np.array([np.sqrt(0.5),0,0,0,np.sqrt(0.5)])}

def base_2_mat(blist,spin):
    
    '''
    Takes in a list of orbital strings, formatted as 'nlm' with m not the ml but a sub-index for the different cubic harmonics.
    n is the index of the atom, moreso than principal quantum number, and l is the orbital angular momentum. so for first atom, 
    dxy state, use '02xy' for example. This operation depends on there being a full set of 2l*1 orbitals for each l given.
    If your model has spin degeneracy, than this should also be included as a Boolean flag. Then the orbital basis will be doubled
    accordingly.
    
    args:
        blist: list of strings, formatted as above
        spin: boolean, dictate whether or not to double the array
    return:
        o2Y: numpy array of dim len(blist)xlen(blist) (or doubled if spin==True) which can be used to rotate a vector 
        in cubic harmonics into a vector in spherical harmonics.
    
    '''
    
    if spin:
        blist+=blist #double the orbital list if using spin
    
    o2Y = np.zeros((len(blist),len(blist)),dtype=complex)
    
    Ynow=[blist[0]] #the user will not necessarily have some standard ordering of their orbitals. This list is populated to the user's choice
    starter = 0 #placekeeper for where to put the subarrays into the o2Y array
    
    for o in range(1,len(blist)):
        if float(blist[o][0])==float(blist[o-1][0]): #if looking at orbital from same shell, add to list
            Ynow.append(blist[o])
        else:
            tmp_o2Y = np.array([Y[Ynow[i][1:]] for i in range(len(Ynow))]).T #generate the cubic to Ylm for this shell
            o2Y[starter:starter+len(Ynow),starter:starter+len(Ynow)] = tmp_o2Y #fill into the larger array
            starter = o #increment the placeholder
            Ynow = [blist[o]] #re-initialize the shell list
            
    tmp_o2Y = np.array([Y[Ynow[i][1:]] for i in range(len(Ynow))]).T #fill in the last sub-array
    o2Y[starter:,starter:] = tmp_o2Y   #fill into the larger array
    return o2Y





if __name__=="__main__":
    
    blist = ["01x","01y","01z","11y","11z","11x"]
    
    Ymat = base_2_mat(blist,False)
            
            


