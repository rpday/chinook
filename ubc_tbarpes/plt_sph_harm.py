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
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import ubc_tbarpes.Ylm as Ylm
import matplotlib.tri as mtri

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


def plot_psi(n,basis,vec):
    th = np.linspace(0,np.pi,2*n)
    ph = np.linspace(0,2*np.pi,2*n)
    th,ph = np.meshgrid(th,ph)
    th,ph = th.flatten(),ph.flatten()
    r = np.zeros((len(basis),len(th)),dtype=complex)
    x = np.zeros((len(basis),len(th)))
    y = np.zeros((len(basis),len(th)))
    z = np.zeros((len(basis),len(th)))
    tri = mtri.Triangulation(th,ph)
    cols = []
    
    max_val = vec[np.where(abs(vec)==abs(vec).max())[0][0]]
    vec*=np.conj(max_val)/abs(max_val)
    
    print(np.around(vec,3))
    
    for ov in list(enumerate(vec)):
        r[ov[0]] = np.around(np.sum(np.array([Ylm.Y(pi[2],pi[3],th,ph)*(pi[0]+1.0j*pi[1])*ov[1] for pi in basis[ov[0]].proj]),axis=0),4)#ov[1]*np.sum(np.array([Ylm.Y(pi[2],pi[3],th,ph)*(pi[0]+1.0j*pi[1]) for pi in basis[ov[0]].proj]),axis=0)
#        print(r[ov[0]].max(),r[ov[0]].min())
#        if abs(r[ov[0]]).max()<1e-4:
#            r[ov[0]] = 2*np.imag(ov[1]*np.sum(np.array([Ylm.Y(pi[2],pi[3],th,ph)*(pi[0]+1.0j*pi[1]) for pi in basis[ov[0]].proj]),axis=0))
        x[ov[0]] = 5*abs(r[ov[0]])**2*np.cos(ph)*np.sin(th)+basis[ov[0]].pos[0]
        y[ov[0]] = 5*abs(r[ov[0]])**2*np.sin(ph)*np.sin(th)+basis[ov[0]].pos[1]
        z[ov[0]] = 5*abs(r[ov[0]])**2*np.cos(th)+basis[ov[0]].pos[2]
#        cols.append(r[ov[0]][tri.triangles][:,1])
        cols.append(col_phase(r[ov[0]][tri.triangles][:,1]))
    
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    for pi in range(len(basis)):
        pplt = ax.plot_trisurf(x[pi],y[pi],z[pi],triangles=tri.triangles,cmap=cm.hsv,antialiased=True,lw=0.1,edgecolors='w')
        pplt.set_array(cols[pi])
        pplt.set_clim(-np.pi,np.pi)
        
        
    ax.set_zlim(-1,1.5)
    cbar = plt.colorbar(pplt,ax=ax)
    cbar.set_label('Phase',rotation=270)
    cbar.ax.locator_params(nbins=3)
    cbar.ax.set_yticklabels(['$\pi$','0','-$\pi$'])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.grid(False)
#    p.set_clim(-0.7,0.7)
    
    return x,y,z,tri.triangles,r,cols




def plot_orbital(n,proj):#basis,vec):
    th = np.linspace(0,np.pi,2*n)
    ph = np.linspace(0,2*np.pi,2*n)
    th,ph = np.meshgrid(th,ph)
    th,ph = th.flatten(),ph.flatten()

    r = np.sum(np.array([Ylm.Y(pi[2],pi[3],th,ph)*(pi[0]+1.0j*pi[1]) for pi in proj]),axis=0)
    x = 5*abs(r)**2*np.cos(ph)*np.sin(th)
    y = 5*abs(r)**2*np.sin(ph)*np.sin(th)
    z = 5*abs(r)**2*np.cos(th)
    tri = mtri.Triangulation(th,ph)
#    cols = r[tri.triangles][:,1]
    cols = col_phase(r[tri.triangles][:,1])
#    cols =rgb_vals(r[tri.triangles][:,1])
    fig = plt.figure(figsize=plt.figaspect(1)*2)
    ax = fig.add_subplot(111,projection='3d')
    p = ax.plot_trisurf(x,y,z,triangles=tri.triangles,cmap=cm.hsv,antialiased=True,edgecolors='w',linewidth=0.2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    p.set_array(cols)
    p.set_clim(-np.pi,np.pi)
    
    return x,y,z,tri.triangles,cols,r


def col_phase(vals):
    x,y=np.real(vals),np.imag(vals)
    return np.arctan2(y,x)
#    sgn = signvec(x)*signvec(y)
#    return np.arctan2(abs(y),sgn*abs(x))


#def sign(x):
#    if x>=0:
#        return 1
#    else:
#        return -1
#    
#signvec = np.vectorize(sign)


#if __name__=="__main__":
#    _,_,_,_,cols,r = plot_orbital(40,np.array([[0,1,1,0]]))
##    psi_dict = {0:np.array([[-np.sqrt(0.5),0,2,1],[np.sqrt(0.5),0,2,-1]]),1:np.array([[0,np.sqrt(0.5),2,1],[0,np.sqrt(0.5),2,-1]]),
#             2:np.array([[-np.sqrt(0.5),0,2,1],[np.sqrt(0.5),0,2,-1]]),3:np.array([[0,np.sqrt(0.5),2,1],[0,np.sqrt(0.5),2,-1]])}
#    k = np.linspace(0,1,2)
##    k = 0
#    
#    psi = np.array([k/2,np.sqrt(1-k**2)*1.0j/2,-np.sqrt(1-k**2)*1.0j/2,k/2])
##    
#    n = 40
#    for ki in range(len(k)):
#        sz = gen_psi(n,psi[:,ki],psi_dict)
        
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