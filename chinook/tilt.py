#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 13:33:59 2018

@author: ryanday
"""

import numpy as np
from collections import Iterable
import matplotlib.pyplot as plt


hb = 6.626e-34/(2*np.pi)
kb = 1.38e-23
q = 1.602e-19
me = 9.11e-31
A = 1e-10


def k_parallel(ek):
    '''
    Convert kinetic energy in eV to inverse Angstrom
    '''
    return np.sqrt(2*me/hb**2*ek*q)*A


def ang_mesh(N,th,ph):
    '''
    Generate a mesh over the indicated range of theta and phi,
    with N elements along each of the two directions
    
    *args*:

        - **N**: int or iterable of length 2 indicating # of points along *th*, and *ph* respectively
        
        - **th**: iterable length 2 of float (endpoints of theta range)
        
        - **ph**: iterable length 2 of float (endpoints of phi range)
    
    *return*:

        - numpy array of N_th x N_ph float, representing mesh of angular coordinates
    
    ***
    '''
    try:
        if isinstance(N,Iterable):
            N_th,N_ph = int(N[0]),int(N[1])
        else:
            N_th,N_ph = int(N),int(N)
    except ValueError:
        print('ERROR: Invalid datatype for number of points in mesh. Return none')
        return None
    return np.meshgrid(np.linspace(th[0],th[1],N_th),np.linspace(ph[0],ph[1],N_ph))

def k_mesh(Tmesh,Pmesh,ek):
    '''
    Application of rotation to a normal-emission vector (i.e. (0,0,1) vector)
    Third column of a rotation matrix formed by product of rotation about vertical (ky), and rotation around kx axis
    c.f. Labbook 28 December, 2018
    
    *args*:

        - **Tmesh**: numpy array of float, output of *ang_mesh*
        
        - **Pmesh**: numpy array of float, output of *ang_mesh*
        
        - **ek**: float, kinetic energy in eV
    
    *return*:

        - **kvec**: numpy array of float, in-plane momentum array associated with angular emission coordinates
    
    ***
    '''
    klen = k_parallel(ek)
    kvec = klen*np.array([-np.sin(Tmesh),-np.cos(Tmesh)*np.sin(Pmesh),np.cos(Tmesh)*np.cos(Pmesh)])

    return kvec



def rot_vector(vector,th,ph):
    '''
    Rotation of vector by theta and phi angles, about the global y-axis by theta, followed by a rotation about
    the LOCAL x axis by phi. This is analogous to the rotation of a cryostat with a vertical-rotation axis (theta),
    and a sample-mount tilt angle (phi). NOTE: need to extend to include cryostats where the horizontal rotation axis
    is fixed, as opposed to the vertical axis--I have never seen such a system but it is of course totally possible.
    
    *args*:

        - **vector**: numpy array length 3 of float (vector to rotate)
        
        - **th**: float, or numpy array of float -- vertical rotation angle(s)
        
        - **ph**: float, or numpy array of float -- horizontal tilt angle(s)
    
    *return*:

        - numpy array of float, rotated vectors for all angles: shape 3 x len(ph) x len(th)
        NOTE: will flatten any length-one dimensions  
    
    ***
    '''
   
    if not isinstance(th,Iterable):
        th = np.array([th])
    if not isinstance(ph,Iterable):     
        ph = np.array([ph])
    th,ph = np.meshgrid(th,ph)
     
    Rm = np.array([[np.cos(th),np.zeros(np.shape(th)),np.sin(th)],[np.sin(th)*np.sin(ph),np.cos(ph),-np.cos(th)*np.sin(ph)],[-np.sin(th)*np.cos(ph),np.sin(ph),np.cos(th)*np.cos(ph)]])
    return np.squeeze(np.einsum('ijkl,j->ikl',Rm,vector))


def plot_mesh(ek,N,th,ph):
    '''
    Plotting tool, plot all points in mesh, for an array of N angles,
    at a fixed kinetic energy.

    *args*:

        - **ek**: float, kinetic energy, eV
        
        - **N**: tuple of 2 int, number of points along each axis
        
        - **thx**: tuple of 2 float, range of horizontal angles, radian
        
        - **thy**: tuple of 2 float, range of vertical angles, radian

    ***
    '''

    kpts = gen_kpoints(ek,N,th,ph,0)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(kpts[:,0],kpts[:,1])
    ax.set_aspect(1)


def gen_kpoints(ek,N,thx,thy,kz):
    '''
    Generate a mesh of kpoints over a mesh of emission angles.
    
    *args*:

        - **ek**: float, kinetic energy, eV
        
        - **N**: tuple of 2 int, number of points along each axis
        
        - **thx**: tuple of 2 float, range of horizontal angles, radian
        
        - **thy**: tuple of 2 float, range of vertical angles, radian
        
        - **kz**: float, k-perpendicular of interest, inverse Angstrom
        
    *return*:

        - **k_array**: numpy array of N[1]xN[0] float, corresponding to mesh of in-plane momenta
        
    ***    
    '''
    Thx,Thy = ang_mesh(N,thx,thy)
    kv = k_mesh(Thx,Thy,ek)
    kx = kv[0,:,:].flatten()
    ky = kv[1,:,:].flatten()
    k_array = np.array([[kx[i],ky[i],kz] for i in range(len(kx))])
    return k_array


if __name__ == "__main__":
    rad = np.pi/180
    kp = k_parallel(60)
    N = 37
    To = 0
    Tlims = [-13*rad+To*rad,13*rad+To*rad]
    Plims = [-25*rad,25*rad]
    #plot_mesh(10,N,Tlims,Plims)
    
    kp = gen_kpoints(21,(20,30),(-1,0),(0,3),0)
    fig  = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(kp[:,0],kp[:,1])