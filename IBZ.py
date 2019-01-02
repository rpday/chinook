#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 08:47:03 2018

@author: ryanday

Generate the irreducible Brillouin zone for a lattice with a set of symmetry operations
from a mesh of k-points

"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ubc_tbarpes.symmetry as symm_tools
import ubc_tbarpes.klib as klib


def _gen_pts(N,lims):
    x = np.linspace(-abs(lims),abs(lims),N)
    X,Y,Z = np.meshgrid(x,x,x)
    X,Y,Z = X.flatten(),Y.flatten(),Z.flatten()
    
    pt_dict = {(np.around(X[i],4),np.around(Y[i],4),np.around(Z[i],4)):1 for i in range(len(X))}
    return pt_dict

def tup_pt(vec):
    return (np.around(vec[0],4),np.around(vec[1],4),np.around(vec[2],4))

def _select_min(v1,v2):
    '''
    Which point to keep? Select principal direction as (1,0,0). Then if one is closer to x axis, keep it.
    If equidistant from x-axis, keep the one in the most positive octant of the 3-sphere.
    args:
        v1--vector for point 1, as list,np.array or tuple of float (len(3))
        v2--vector for point 2, as list,np.array or tuple of float (len(3))
    '''
    c,n = v2,v1
    if np.sqrt(v1[1]**2+v1[2]**2)>np.sqrt(v2[1]**2+v2[2]**2):
        n,c = v2,v1
    else:
        sgn_1 = np.sign(v1[0])+np.sign(v1[1])+np.sign(v1[2])
        sgn_2 = np.sign(v2[0])+np.sign(v2[1])+np.sign(v2[2])
        if sgn_2>sgn_1:
            n,c = v2,v1
    return c,n
            
    


def op_reduce(op_args,pt_dict):
    '''
    Eliminate duplicate points in k-space after symmetry operation. Iterate through list of 
    operations, and following each, check for duplicates in the k-mesh
    args:
        op_args -- list of tuples, each containing the symmetry operation to be performed
        pt_dict -- dictionary of tuples corresponding to the location of k-mesh points
    return:
        pt_dict -- reduced dictionary of k-points, along with the multiplicity of a given point 
    '''
    for o in op_args:
        tmp = np.array([p for p in pt_dict])
        tmp_p = np.around(symm_tools._operate_(o,tmp),4)
        for t in range(len(tmp_p)):
            if tup_pt(tmp_p[t]) in pt_dict.keys() and tup_pt(tmp[t]) in pt_dict.keys():
                if (tmp[t,0]==tmp_p[t,0]) and (tmp[t,1]==tmp_p[t,1]) and (tmp[t,2]==tmp_p[t,2]):
                    continue
                else:
                    c,n = _select_min(tup_pt(tmp_p[t]),tup_pt(tmp[t]))
                    pt_dict[tup_pt(n)]+=pt_dict[tup_pt(c)]
                    _= pt_dict.pop(tup_pt(c))
                    

    return pt_dict




            
def _d_to_arr(pt_dict):
    pts = np.array([[p[0],p[1],p[2]] for p in pt_dict])
    return pts

def _plt_pts(pt_arr,lims):
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(pt_arr[:,0],pt_arr[:,1],pt_arr[:,2])
    ax.set_xlim(-abs(lims),abs(lims))
    ax.set_ylim(-abs(lims),abs(lims))
    ax.set_zlim(-abs(lims),abs(lims))
    
    
#def BZ_mesh(avecs):
    
    
    

if __name__=="__main__":
    
    N,lims = 7, 6
    args = (('M',np.array([1,1,0])),('M',np.array([1,-1,0])),('M',np.array([0,1,0])),('M',np.array([1,0,0])),('M',np.array([0,0,1])),('R',np.array([1,1,1]),np.pi/3),('R',np.array([1,1,1]),2*np.pi/3),('R',np.array([0,0,1]),np.pi/2),('R',np.array([0,0,1]),np.pi),('R',np.array([0,0,1]),3*np.pi/2))
    args = (('M',np.array([0,0,1])),('R',np.array([0,0,1]),np.pi/3),('R',np.array([0,0,1]),2*np.pi/3),('R',np.array([0,0,1]),np.pi/2),('R',np.array([0,0,1]),4*np.pi/3),('R',np.array([0,0,1]),5*np.pi/3),('M',np.array([1,0,0])),('M',np.array([np.sqrt(3),1,0])),('M',np.array([-np.sqrt(3),1,0])),('M',np.array([0,1,0])))
    BZ = _gen_pts(N,lims)
    IBZ = op_reduce(args,BZ)
    IBZ_pts = _d_to_arr(IBZ)
    _plt_pts(IBZ_pts,lims)
    
    
            
    
            
    

