# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 12:37:56 2018

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

'''
Plotting orbital along a path in momentum and energy


'''


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import ubc_tbarpes.Ylm as Ylm
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D

def gen_psi(n,psi,psi_dict,str_nm=None):
    th = np.linspace(0,np.pi,2*n)
    ph = np.linspace(0,2*np.pi,2*n)
    th,ph = np.meshgrid(th,ph)
    th,ph = th.flatten(),ph.flatten()
    psi/=np.dot(np.conj(psi),psi)
    
    psi_d  = np.sum(np.array([Ylm.Y(pi[2],pi[3],th,ph)*(pi[0]+1.0j*pi[1])*psi[p] for p in range(int(len(psi)/2)) for pi in psi_dict[p]]),0)    
    psi_u  = np.sum(np.array([Ylm.Y(pi[2],pi[3],th,ph)*(pi[0]+1.0j*pi[1])*psi[p] for p in range(int(len(psi)/2),len(psi)) for pi in psi_dict[p]]),0)    
    r = abs(psi_u)**2+abs(psi_d)**2
    x = r*np.cos(ph)*np.sin(th)
    y = r*np.sin(ph)*np.sin(th)
    z = r*np.cos(th)
    
    sz = (abs(psi_u)**2-abs(psi_d)**2)/(r)
#    if sum(sz)<=0:
    tri = mtri.Triangulation(th,ph)

    cols = sz[tri.triangles][:,1]
    
    
    fig = plt.figure(figsize=plt.figaspect(1)*2)  
    
    ax = fig.add_subplot(111,projection='3d')
    p = ax.plot_trisurf(x,y,z,triangles=tri.triangles,cmap=cm.RdBu,shade=True,antialiased=True,edgecolors='k',linewidth=0.2)
    p.set_array(cols)
    p.set_clim(-1.2,1.2)
    ax._axis3don=False
    if str_nm is not None:
        plt.savefig(str_nm,transparent=True)

    return sz


if __name__=="__main__":
    psi_dict = {0:np.array([[-np.sqrt(0.5),0,2,1],[np.sqrt(0.5),0,2,-1]]),1:np.array([[0,np.sqrt(0.5),2,1],[0,np.sqrt(0.5),2,-1]]),
             2:np.array([[-np.sqrt(0.5),0,2,1],[np.sqrt(0.5),0,2,-1]]),3:np.array([[0,np.sqrt(0.5),2,1],[0,np.sqrt(0.5),2,-1]])}
    k = np.linspace(0,1,2)
#    k = 0
    
    psi = np.array([k/2,np.sqrt(1-k**2)*1.0j/2,-np.sqrt(1-k**2)*1.0j/2,k/2])
#    
    n = 40
    for ki in range(len(k)):
        sz = gen_psi(n,psi[:,ki],psi_dict)
        
#    d1d,d1u = {o.index:o.proj for o in TB.basis[:5]},{(o.index-5):o.proj for o in TB.basis[10:15]}
#    d2d,d2u = {(o.index-5):o.proj for o in TB.basis[5:10]},{(o.index-10):o.proj for o in TB.basis[15:]}
#    d1 = {**d1d,**d1u}
#    d2 = {**d2d,**d2u}
#    ind =11
##    for oind in range(7,13):
##        ind = oind
#    psi_11 = np.array([list(TB.Evec[i,:5,ind])+list(TB.Evec[i,10:15,ind]) for i in range(len(Kobj.kpts))])
#    
#    psi_12 = np.array([list(TB.Evec[i,5:10,ind])+list(TB.Evec[i,15:,ind]) for i in range(len(Kobj.kpts))])
#       
#    #    psi_1 = np.array([list(TB.Evec[i,:,ind+2]) for i in range(len(Kobj.kpts))])
#    #    df = {o.index:o.proj for o in TB.basis}
#    #    psi_11 = np.array([list(TB.Evec[i,:5,ind:ind+2])+list(TB.Evec[i,10:15,ind:ind+2]) for i in range(len(Kobj.kpts))])
#        
#    for ki in range(len(Kobj.kpts)):
#        tmp =ki
##        strnm = 'C:\\Users\\rday\\Documents\\TB_ARPES\\2018\\TB_ARPES_2018\\plots\\FeSe_kz\\no_SO_{:d}_{:0.1f}Z.png'.format(ind,tmp)
#        sph.gen_psi(30,psi_11[ki],d1)#strnm)