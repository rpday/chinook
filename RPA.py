# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 21:56:29 2018
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
import ubc_tbarpes.klib as K_lib



'''
This script is intended to compute the RPA susceptibility in spin and charge channels for FeSCs.
Begin by computing bare susceptibility

Then go on to compute the interacting, RPA approximation to this

'''



def chi_0(TB,bvec,Nk,T,lvals):
     
    
    kx = np.linspace(-bvec[0]*1,bvec[0]*1,Nk)
    ky = np.linspace(-bvec[1]*1,bvec[1]*1,Nk)
    Kx,Ky= np.meshgrid(kx,ky)
    qx = np.linspace(-bvec[0]*0.5,bvec[0]*0.5,int(Nk/2))
    qy =np.linspace(-bvec[1]*0.5,bvec[1]*0.5,int(Nk/2))
    Qx,Qy= np.meshgrid(qx,qy)
    
    k_arr,_ = K_lib.kmesh(0,Kx,Ky,np.zeros(np.shape(Kx)))      
    
    TB.Kobj = K_lib.kpath(k_arr)
    TB.Eb,TB.Ev = TB.solve_H()
    TB.Eb=np.reshape(TB.Eb,(Nk,Nk,5))
    TB.Ev = np.reshape(TB.Ev,(Nk,Nk,5,5))
    fv = fermi(TB.Eb,T)
    
    
    def X_raw_gen(ki,kj,u,v,i,j):
        return TB.Ev[ki,kj,lvals[3],u]*np.conj(TB.Ev[ki,kj,lvals[1],u])*TB.Ev[ki+i,kj+j,lvals[0],v]*np.conj(TB.Ev[ki+j,kj+j,lvals[2],v])*(fv[ki+i,kj+j,v]-fv[ki,kj,u])/(TB.Eb[ki+i,kj+j,v]-TB.Eb[ki,kj,u])          
     
    
    u =np.linspace(0,4,5).astype(int)
    v =np.linspace(0,4,5).astype(int)
    isub = np.linspace(int(Nk/4),int(Nk*3/4)-1,int(Nk/2)).astype(int)
    jsub = isub
    Qi = np.linspace(-int(Nk/4),int(Nk/4)-1,int(Nk/2)).astype(int)
    Qj = Qi
    Ki,Kj,U,V=np.meshgrid(isub,jsub,u,v)
    X_out = np.zeros((np.shape(Qx)),dtype=complex)
    for i in Qi:
        for j in Qj:
            
            
            X_out[i+Qi[0],j+Qj[0]] =-1.0/Nk**2*np.sum(np.sum(np.sum(np.sum(X_raw_gen(Ki,Kj,U,V,i,j),0),0),0),0)
    
    nan_mask=np.where(np.isnan(X_out))
    X_out[nan_mask]=0.0
    
    
    fig = plt.figure()
    ax = fig.add_subplot(121)
    p = ax.pcolormesh(Qx,Qy,np.real(X_out))
    ax2 =fig.add_subplot(122)
    p2 = ax2.pcolormesh(Qx,Qy,np.imag(X_out))
    
    return X_out
    
           


          


def fermi(E,T):
    return 1./(np.exp(E/T)+1)
    
    
    