#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Created on Mon Nov 13 18:28:22 2017
#@author: ryanday
#MIT License

#Copyright (c) 2018 Ryan Patrick Day

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.


#For now, given a set of pts--numpy array 3xN and a grain --# points between each of the N pts,
#this will generate a Kobject, with labels, grain, and other attributes useful for plotting
#In the future it would be nice to generate a klist based on an xcrysden calculation for example.



import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


me= 9.11e-31
hb = 6.626e-34/(2*np.pi)
q = 1.602e-19
A=1.0e-10

class kpath:
    
    '''
    Momentum object, defining a path in reciprocal space, for use in defining the 
    Hamiltonian at different points in the Brillouin zone.
    ***
    '''
    
    def __init__(self,pts,grain=None,labels=None):
        
        '''
        Initialize the momentum object
        
        *args*:
            - **pts**: list of len(3) numpy array of float: the endpoints of the 
            path in k-space
            
        *kwargs*:
            - **grain**: int, optional, indicating number of points between each point 
            in **pts**
            
            - **labels**: list of strings, optional, labels for plotting bandstructure
            along the kpath
        ***
        '''
        
        if type(pts)==list:
            self.pts = pts
            self.grain = grain
            self.kpts = [] #will be the actual vector k points for calculation
            self.kcut = [] #will be a 1D array for plotting purposes
            self.kcut_brk = []
            self.ind_brk = [0]
            self.labels=labels
            self.points()
        elif type(pts)==np.ndarray:
            self.kpts = pts #if we are just feeding in an array of kpoints
            
    def points(self):
        
        '''
        Use the endpoints of kpath defined in **kpath.pts** to create numpy array
        of len(3) float which cover the entire path, based on method by I.S. Elfimov.
        
        *return*:
            - **kpath.kpts**: numpy array of float, len(**kpath.pts**)(1+**kpath.grain**) by 3
        
        ***
        '''
        
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
    
    
def bvectors(a_vec):
    '''
    Define the reciprocal lattice vectors corresponding to the direct lattice 
    in real space
    
    *args*:
        - **a_vec**: numpy array of 3x3 float, lattice vectors
        
    *return*:
        - **b_vec**: numpy array of 3x3 float, reciprocal lattice vectors
       
    ***
    '''
    b_vec = 2*np.pi*np.array([(np.cross(a_vec[1],a_vec[2])/np.dot(a_vec[0],np.cross(a_vec[1],a_vec[2]))),(np.cross(a_vec[2],a_vec[0])/np.dot(a_vec[1],np.cross(a_vec[2],a_vec[0]))),(np.cross(a_vec[0],a_vec[1])/np.dot(a_vec[2],np.cross(a_vec[0],a_vec[1])))])
    return b_vec


def b_zone(a_vec,N,show=False):
    '''
    Generate a mesh of points over the Brillouin zone. 
    Each of the cardinal axes are divided by the same number of points 
    (so points are not necessarily evenly spaced along each axis).
    
    *args*:
        - **a_vec**: numpy array of size 3x3 float
        
        - **N**: int mesh density
   
    *kwargs*:     
        - **show**: boolean for optional plotting of the mesh points
        
    *return*:
        - **m_pts**: numpy array of mesh points (float), shape (len(m_pts),3)
    '''
    
    b_vec = bvectors(a_vec)
    rlpts = np.dot(region(1),b_vec)    
    bz_mesh_raw = raw_mesh(rlpts,N)
    bz_mesh = mesh_reduce(rlpts,bz_mesh_raw)
    
    if show:
        plt_pts(bz_mesh)

    return bz_mesh



def region(num):
    '''
    Generate a symmetric grid of points in number of lattice vectors.
    
    *args*:
        - **num**: int, grid will have size 2 num+1 in each direction
        
    *return*:
        - numpy array of size ((2 num+1)**3,3) with centre value of 
        first entry of (-num,-num,-num),...,(0,0,0),...,(num,num,num)
        
    ***
    '''
    num_symm = 2*num+1
    return np.array([[int(i/num_symm**2)-num,int(i/num_symm)%num_symm-num,i%num_symm-num] for i in range((num_symm)**3)])



def raw_mesh(blatt,N):
    
    '''
    Define a mesh of points filling the region of k-space bounded by the set
    of reciprocal lattice points generated by *bvectors*.
    These will be further reduced by *mesh_reduce* to find points which
    are within the first-Brillouin zone
    
    *args*:
        - **blatt**: numpy array of 27x3 float
        
        - **N**: int, or iterable of len 3, defines a coarse estimation
        of number of k-points
        
    *return*:
        - **mesh**: numpy array of mesh points, size set roughly by N
        
    ***
    '''
    limits = np.array([[blatt[:,ii].min(),blatt[:,ii].max()] for ii in range(3)])
    if type(N)==int:
        L_max = (limits[:,1]-limits[:,0]).min()
        dk = [(L_max/N)]*3
    elif type(N)==tuple or type(N)==list or type(N)==np.ndarray:
        if len(N)==3:
            dk = [(limits[i,1]-limits[i,0])/N[i] for i in range(3)]
        else:
            print('ERROR: Must pass 3 numbers if passing a list/tuple/array')
            return []
        
    else:
        print('Invalid K division type. Pass only integer or list/tuple/array of 3 integer values')
        return []
    x,y,z = np.arange(limits[0,0],limits[0,1]+dk[0],dk[0]),np.arange(limits[1,0],limits[1,1]+dk[1],dk[1]),np.arange(limits[2,0],limits[2,1]+dk[2],dk[2])
    Y,Z,X = np.meshgrid(y,z,x) #This ordering for the meshgrid allows us to define 1D array with register looping over x, then y, then z. Not sure why this order gives that output...
    X,Y,Z = X.flatten(),Y.flatten(),Z.flatten()
    Kpts = np.array([[X[i],Y[i],Z[i]] for i in range(len(X))])
    return Kpts

def mesh_reduce(blatt,mesh,inds=False):
    '''
    Determine and select only k-points corresponding to the first Brillouin
    zone, by simply classifying points on the basis
    of whether or not the closest lattice point is the origin. 
    By construction, the origin is index 13 of the blatt. 
    If it is not, return error. Option to take only the indices of
    the mesh which we want, rather than the actual array points
    --this is relevant for tetrahedral interpolation methods
    
    *args*:
        - **blatt**: numpy array of len(27,3), nearest reciprocal lattice vector points
        
        - **mesh**: numpy array of (N,3) float, defining a mesh of k points, before
        being reduced to contain only the Brillouin zone.
        
    *kwargs*: 
        - **inds**: option to pass a list of bool, indicating the 
        indices one wants to keep, instead of autogenerating the mesh
    
    *return*:
        - **bz_pts**: numpy array of (M,3) float, Brillouin zone points
    
    '''
    bz_pts = []
    if np.linalg.norm(blatt[13])>0:       
        print('FORMAT ERROR: invalid Reciprocal Lattice Point array passed. Please use np.dot(region(1),b_vec) to generate these points')
        return []
    else:
        for m in list(enumerate(mesh)):
            dv = np.linalg.norm(np.around(m[1]-blatt,4),axis=1)
            if (13 in np.where(dv==dv.min())[0]):
                bz_pts.append(m[not inds])


        return np.array(bz_pts)
    

def plt_pts(pts):
    '''
    Plot an array of points iin 3D
    
    *args*:
        - **pts**: numpy array shape N x 3
    
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(pts[:,0],pts[:,1],pts[:,2])
 
    
    
def kmesh(ang,X,Y,kz,Vo=None,hv=None,W=None):
    '''
    Take a mesh of kx and ky with fixed kz and generate a Nx3 array of points
    which rotates the mesh about the z axis by **ang**. N is the flattened shape
    of **X** and **Y**.
    
    *args*:
        - **ang**: float, angle of rotation
        
        - **X**: numpy array of float, one coordinate of meshgrid
        
        - **Y**: numpy array of float, second coordinate of meshgrid
        
        - **kz**: float, third dimension of momentum path, fixed
        
        
    *kwargs*:
        - **Vo**: float, parameter necessary for inclusion of inner potential
        
        - **hv**: float, photon energy, to be used if **Vo** also included, for evaluating kz
        
        - **W**: float, work function
        
    *return*:
        - **k_arr**: numpy array of shape Nx3 float, rotated  kpoint array.
        
        - **ph**: numpy array of N float, angles of the in-plane momentum
        points, before rotation.
    
    ***    
    '''
            
    kp = np.sqrt(X**2+Y**2)
    ph = np.arctan2(Y,X)
    if abs(ang)>0.0:
        X = kp*np.cos(ph-ang)
        Y = kp*np.sin(ph-ang)
    try:
        Zeff = kz_kpt(hv,kp,W,Vo)
        print('Inner potential detected, overriding kz input.')
    except TypeError:    
        Zeff = kz*np.ones(np.shape(X))
    
    ph = np.reshape(ph,np.shape(X)[0]*np.shape(X)[1])
    k_arr = np.reshape(np.array([X,Y,Zeff]),(3,np.shape(X)[0]*np.shape(X)[1])).T
    return k_arr,ph


def kmesh_hv(ang,x,y,hv,Vo=None):
    '''
    Take a mesh of kx and ky with fixed kz and generate a Nx3 array of points
    which rotates the mesh about the z axis by **ang**. N is the flattened shape
    of **X** and **Y**.
    
    *args*:
        - **ang**: float, angle of rotation
        
        - **X**: numpy array of float, one coordinate of meshgrid
        
        - **Y**: numpy array of float, second coordinate of meshgrid
        
        - **kz**: float, third dimension of momentum path, fixed
        
        
    *kwargs*:
        - **Vo**: float, parameter necessary for inclusion of inner potential
        
        - **hv**: float, photon energy, to be used if **Vo** also included, for evaluating kz
        
        - **W**: float, work function
        
    *return*:
        - **k_arr**: numpy array of shape Nx3 float, rotated  kpoint array.
        
        - **ph**: numpy array of N float, angles of the in-plane momentum
        points, before rotation.
    
    ***    
    '''
            

    
    if len(x)==1:

        X,HV = np.meshgrid(y,hv)
        Y = x*np.ones(np.shape(X))
    elif len(y)==1:
        X,HV = np.meshgrid(x,hv)
        Y = y*np.ones(np.shape(X))
        
        
    ph = np.arctan2(Y,X)
    KP = np.sqrt(X**2+Y**2)    

    if abs(ang)>0.0:
        x = KP*np.cos(ph-ang)
        y = KP*np.sin(ph-ang)
    Z = kz_kpt(HV,KP,V=Vo)
    
    ph = np.reshape(ph,np.shape(X)[0]*np.shape(X)[1])
    k_arr = np.reshape(np.array([X,Y,Z]),(3,np.shape(X)[0]*np.shape(X)[1])).T
    return k_arr,ph


def kz_kpt(hv,kpt,W=0,V=0):
    
    '''
    Extract the kz associated with a given in-plane momentum, photon energy, 
    work function and inner potential
    
    *args*:
        - **hv**: float, photon energ in eV
        
        - **kpt**: float, in plane momentum, inverse Angstrom
        
        - **W**: float, work function in eV
        
        - **V**: float, inner potential in eV
        
    *return*:
        - **kz**: float, out of plane momentum, inverse Angstrom
    ***
    '''
    kn = np.sqrt(2*me/hb**2*(hv-W)*q)*A
    
    kz= np.sqrt(2*me/hb**2*((hv-W)*(1-(kpt/kn)**2)+V)*q)*A
    
    return kz

if __name__=="__main__":
    a = 1.78
    av = np.array([[a,0,a],[0,a,a],[a,a,0]])
#    av = np.array([[np.sqrt(3)*a/2,a/2,0],[np.sqrt(3)*a/2,-a/2,0],[0,0,2*c]])
    N = 20
    bz = b_zone(av,N,True)