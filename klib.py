#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 18:28:22 2017

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
For now, given a set of pts--numpy array 3xN and a grain --# points between each of the N pts,
this will generate a Kobject, with labels, grain, and other attributes useful for plotting
In the future it would be nice to generate a klist based on an xcrysden calculation for example.
'''


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


me= 9.11e-31
hb = 6.626e-34/(2*np.pi)
q = 1.602e-19
A=1.0e-10

class kpath:
    
    def __init__(self,pts,grain=None,labels=None):
        if type(pts)==type([]):
            self.pts = pts
            self.grain = grain
            self.kpts = [] #will be the actual vector k points for calculation
            self.kcut = [] #will be a 1D array for plotting purposes
            self.kcut_brk = []
            self.ind_brk = [0]
            self.labels=labels
            self.points()
        elif type(pts)==type(np.array([])):
            self.kpts = pts #if we are just feeding in an array of kpoints
    def points(self):
        #This method based on I.S. Elfimov function
        cutst = 0.0
        self.kcut_brk.append(cutst)
        self.kpts = [(self.pts[0])]
        self.kcut.append(cutst)
        for p in range(len(self.pts)-1):
            tempvec = np.array(self.pts[p+1])-np.array(self.pts[p])
            vecstep = tempvec/self.grain

            for i in range(1,self.grain+1):
                self.kpts.append(list(self.pts[p]+vecstep*float(i)))
                self.kcut.append(cutst+np.linalg.norm(vecstep)*float(i))
            cutst = self.kcut[-1]
            self.kcut_brk.append(cutst)
            self.ind_brk.append(len(self.kcut))
        self.kpts = np.array(self.kpts)
        return self.kpts
    
###ADD MESH GENERATION FOR e.g. DOS type-calculations    
    
def bvectors(a_vec):
    b_vec = 2*np.pi*np.array([(np.cross(a_vec[1],a_vec[2])/np.dot(a_vec[0],np.cross(a_vec[1],a_vec[2]))),(np.cross(a_vec[2],a_vec[0])/np.dot(a_vec[1],np.cross(a_vec[2],a_vec[0]))),(np.cross(a_vec[0],a_vec[1])/np.dot(a_vec[2],np.cross(a_vec[0],a_vec[1])))])
    return b_vec


def b_zone(a_vec,N,show=False):
    '''
    Generate a mesh of points over the Brillouin zone
    The mesh will be span the region of points enclosed by the reciprocal lattice vectors G=(-1,-1,-1) .. G(1,1,1)
    dividing each of the cardinal axes by the same number of points (so points are not necessarily evenly spaced).
    Instead each axis of the zone is considered equally. 
    args:
        a_vec -- numpy array of size 3x3 float
        N -- int mesh density
        show -- boolean for optional plotting of the mesh points
    return:
        m_pts -- numpy array of mesh points (float), shape (len(m_pts),3)
    '''
    b_vec = bvectors(a_vec)
    nn = 3
    rlpts = np.array([np.dot(b_vec,np.array([(int(i/nn**2)-int(nn/2)),(int(i/nn)%nn-int(nn/2)),i%nn-int(nn/2)])) for i in range(nn**3) if i!=int(nn**3/2)])    
    
    x = np.linspace(-abs(rlpts[:,0]).max()*0.6,abs(rlpts[:,0]).max()*0.6,N*2+1)
    y = np.linspace(-abs(rlpts[:,1]).max()*0.6,abs(rlpts[:,1]).max()*0.6,N*2+1)
    z = np.linspace(-abs(rlpts[:,2]).max()*0.6,abs(rlpts[:,2]).max()*0.6,N*2+1)
    X,Y,Z = np.meshgrid(x,y,z)
    X,Y,Z = X.flatten(),Y.flatten(),Z.flatten()
    d_pt = np.array([np.sqrt((X[i]-rlpts[:,0])**2+(Y[i]-rlpts[:,1])**2+(Z[i]-rlpts[:,2])**2) for i in range(len(X))])
#    return d_pt
#    print(np.shape(d_pt))
    m_pts = np.array([[X[i],Y[i],Z[i]] for i in range(len(X)) if d_pt[i].min()>=np.sqrt(X[i]**2+Y[i]**2+Z[i]**2)])
#    
    if show:
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        ax.scatter(m_pts[:,0],m_pts[:,1],m_pts[:,2])
#        ax.scatter(rlpts[:,0],rlpts[:,1],rlpts[:,2],c='r')
        plt.show()

    return m_pts
    


def kmesh(ang,X,Y,kz):
    '''
    Take a mesh of kx and ky with fixed kz and generate a 3xN array of points
    '''
            
    kp = np.sqrt(X**2+Y**2)
    ph = np.arctan2(Y,X)
    if abs(ang)>0.0:
        X = kp*np.cos(ph-ang)
        Y = kp*np.sin(ph-ang)
    Zeff = kz*np.ones(np.shape(X))
    
    ph = np.reshape(ph,np.shape(X)[0]*np.shape(X)[1])
    k_arr = np.reshape(np.array([X,Y,Zeff]),(3,np.shape(X)[0]*np.shape(X)[1])).T
    return k_arr,ph


def kz_kpt(hv,kpt,W,V):
    kn = np.sqrt(2*me/hb**2*(hv-W)*q)*A
    
    kz= np.sqrt(2*me/hb**2*((hv-W)*(1-(kpt/kn)**2)+V)*q)*A
    
    return kz

if __name__=="__main__":
    a,c=2.46,3.35
    av = np.array([[a,0,a],[0,a,a],[a,a,0]])
    av = np.array([[np.sqrt(3)*a/2,a/2,0],[np.sqrt(3)*a/2,-a/2,0],[0,0,2*c]])
    N = 10
    bz = b_zone(av,N,True)