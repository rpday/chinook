#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 20:23:57 2019

@author: ryanday
"""

import numpy as np
import chinook.rotation_lib as rotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm




def arc(a,N):
    
    phi = np.linspace(0,2*np.pi,N)
    pts = np.array([np.sin(a)*np.cos(phi),0.6*np.sin(a)*np.sin(phi),np.zeros(N)]).T
    
    return pts


def rotate_pts(vecs,pts,axis,angle):
    
    Rmat = rotlib.Rodrigues_Rmat(axis,angle)
    pts2 = np.dot(Rmat,pts.T).T
    vecs2 = np.einsum('ij,klj->kli',Rmat,vecs)
    return pts2,vecs2



def plot_arc(vecs,pts):
    fig = plt.figure()
    ax3 = fig.add_subplot(121,projection='3d')
    ax2 = fig.add_subplot(122)
    ax3.plot(vecs[:,0,0],vecs[:,0,1],vecs[:,0,2])
    ax3.plot(vecs[:,1,0],vecs[:,1,1],vecs[:,1,2])
    ax3.scatter(pts[:,0],pts[:,1],pts[:,2])
    ax2.scatter(pts[:,0],pts[:,1])
    ax3.set_xlim(-1,1)
    ax3.set_ylim(-1,1)
    ax3.set_zlim(-1,1)
    ax3.set_aspect(1)
    ax2.set_aspect(1)
    
    return pts

def rotate_pol(vec,slit,fix,scan):
    if slit.upper()=='H':
        R1 = rotlib.Rodrigues_Rmat(np.array([0,1,0]),fix)
        R2 =  np.array([rotlib.Rodrigues_Rmat(np.array([1,0,0]),scan[i]) for i in range(len(scan))])
    elif slit.upper() == 'V':
        R1 = rotlib.Rodrigues_Rmat(np.array([1,0,0]),fix)
        R2 = R2 = np.array([rotlib.Rodrigues_Rmat(np.array([0,np.cos(fix),np.sin(fix)]),scan[i]) for i in range(len(scan))])

#        R2 = np.array([rotlib.Rodrigues_Rmat(np.array([0,1,0]),scan[i]) for i in range(len(scan))])
    
    vec0 = np.dot(R1,vec)
    print(vec0)
    vec1 = np.einsum('ijk,k->ij',R2,vec0)
    
    return vec1


def plot_pols(vec,vec2):
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    vec = np.array([[0,0,0],vec])
    ax.plot(vec[:,0],vec[:,1],vec[:,2])
    colors = cm.magma(np.linspace(0,1,len(vec2)))
    for ii in range(len(vec2)):
        vec2i = np.array([[0,0,0],vec2[ii]])
        ax.plot(vec2i[:,0],vec2i[:,1],vec2i[:,2],c = colors[ii])
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1,1)
    ax.set_aspect(1)
    


if __name__ == "__main__":
    
    a = np.pi/6
    N = 20
    pts = arc(a,N)
    vecs = np.array([[[0,0,0],[0,0,0],[0,0,0],[0,0,0]],[[0,0,1],[np.sqrt(0.5),0,np.sqrt(0.5)],[1,0,0],[0,1,0]]])
    
    axis = np.array([1,0,0])
    angle = np.pi/20
    
    axis2 = np.array([0,1,0])
    angle2 = np.pi/4
    
#    pts2,vecs2 = rotate_pts(vecs,pts,axis,angle)
#    
##    axis2 = vecs2[1,2,:]
#    pts3,vecs3 = rotate_pts(vecs2,pts2,axis2,angle2)
#    
#    plot_arc(vecs,pts)
#    plot_arc(vecs2,pts2)
#    plot_arc(vecs3,pts3)
#    
#    
#    
    
    vec0 = vecs[1,1,:]
    fix = np.pi/2
    scan = np.linspace(0,-np.pi/4,10)
    rotated = rotate_pol(vec0,'V',fix,scan)
    plot_pols(vec0,rotated)
    