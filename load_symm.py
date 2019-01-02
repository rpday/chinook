#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 16:56:30 2018

@author: ryanday
"""

import sys
#sys.path.append('')

import pkg_resources
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ubc_tbarpes.symmetry as symm_tools
import ubc_tbarpes.klib as klib



textnm="space_group_operations.txt"
filename = pkg_resources.resource_filename(__name__,textnm)

def find_group(number):
    '''
    Find the group associated with the space group number passed.
    Grab all symmetry operations associated with the space group,
    to be used in defining the irreducible Brillouin zone
    args:
        number -- int (from 1-230 inclusive) index of the space group
    return:
        list of lists of strings indicating all related symmetry operations
        for this space group
    
    '''
    if number<1 or number>230:
        print('ERROR: Invalid space group passed--only identity will be taken as a symmetry operation)')
        return [['X','Y','Z']]
    found = False
    operations = []
    with open(filename,'r') as origin:
        for line in origin:
            if '#{:d}'.format(number) in line:
                print('You have selected {:s}'.format(line))
                found = True
                continue
            if found:
                parse = [l.strip() for l in line.split(',')]
                if len(parse)>1:
                    operations.append(parse)
                elif len(parse)==1:
                    found = False
                    break
    return operations
                    

def read_operation(op_list):
    
    '''
    Take the list of strings indicating the symmetry operation and return a 4x4 matrix associated with 
    the operation. (4th dimension is necessary for translation)   
    '''
    X = np.array([1,0,0,0])
    Y = np.array([0,1,0,0])
    Z = np.array([0,0,1,0])
    T = np.array([0,0,0,1])
    omat = np.zeros((4,4),dtype=float)
    omat[-1,-1] = 1.0
    for oi in range(len(op_list)):
        phrase = '+-1*'.join(op_list[oi].split('-'))
        psplit = phrase.split('+')
        for o in range(len(psplit)):
            try:
                if not psplit[o][-1].isalpha(): #an isolated numeric element indicates a translation, find all such elements
                    psplit[o] = psplit[o]+'*T'
            except IndexError:
                next
        phrase = '+'.join(psplit)
        omat[oi] = eval(phrase)
    return omat


def _operate_(U,vecs):
    '''
    Perform symmetry operation on an array of vectors. To accommodate translations,
    all symmetry operations act on an artificial 4 dimensional space, since the translation
    matrix is represented as a 4x4 matrix. 
    The user sends a tuple organized with a single character to indicate the operation type:
        T -- translation, R -- rotation, M -- mirror, I -- inversion, G -- glide
    The other elements of the tuple are the corresponding arguments, e.g. for glide, these will
    be the mirror plane and translation vector, ('G',np.array([0,0,1]),np.array([0.5,0.5,0.0]))
    Composite operations can also be defined, in which case we iterate over each tuple in a list of
    tuples
    args:
        args -- tuple of args, or tuple of tuple of args to seed generation of the transformation matrices
                -- vecs -- 3 x N array of float of vectors to be transformed ( can also pass N x 3 and will reshape to Transpose. Note this is not safe against pathological case of 3x3!!)
    return: transformed vectors, represented as 3xN array of float (or N x 3 if that's what's passed in)
    '''
    
    v_4d = np.zeros((4,len(vecs)))
    less = True
    try:
        v_4d[:3,:] = vecs
    except ValueError:
        less = False
        v_4d[:3,:] = vecs.T
    
    v_4d[-1,:]=1
   
    
    if less:   
        return np.dot(U,v_4d)[:3,:]
    else:
        return np.dot(U,v_4d)[:3,:].T 
    
    


def _gen_pts(N):
    x = np.arange(-0.5,0.5+1/N,1/N)
#    x = np.linspace(-abs(lims),abs(lims),N)
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


def rotate_mats(mats,avec):
    avi = np.identity(4)
    av = np.identity(4)
    avi[:3,:3] = np.linalg.inv(avec).T
    av[:3,:3] = avec.T
    new_mats = np.zeros(np.shape(mats))
    for m in list(enumerate(mats)):
        new_mats[m[0]] = np.dot(av,np.dot(m[1],avi))
    return new_mats
    
    
    
def op_reduce(op_mats,pt_dict):
    '''
    
    Eliminate duplicate points in k-space after symmetry operation. Iterate through list of 
    operations, and following each, check for duplicates in the k-mesh
    args:
        op_mats -- list of 4x4 numpy arrays, each containing the symmetry operation to be performed
        pt_dict -- dictionary of tuples corresponding to the location of k-mesh points
    return:
        pt_dict -- reduced dictionary of k-points, along with the multiplicity of a given point 
    '''
    for o in op_mats:
        tmp = np.array([p for p in pt_dict]) #original points
        tmp_p = np.around(_operate_(o,tmp),4) #operated points
        for t in range(len(tmp_p)): # for each operated point
            if tup_pt(tmp_p[t]) in pt_dict.keys() and tup_pt(tmp[t]) in pt_dict.keys(): #if original and ro
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

def rotate_points(points,avec):
    mat = avec
    points_2 = np.dot(points,mat)
    return points_2

def _plt_pts(pt_arr,lims):
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(pt_arr[:,0],pt_arr[:,1],pt_arr[:,2])
    ax.set_aspect(1)
#    ax.set_xlim(-abs(lims),abs(lims))
#    ax.set_ylim(-abs(lims),abs(lims))
#    ax.set_zlim(-abs(lims),abs(lims))
        

if __name__ == "__main__":
    a,c= 5,10
    avec = np.array([[a,0,0],[-a/2,a*np.sqrt(3)/2,0],[0,0,c]])
    
    num = 143
    ops = find_group(num)
    mats = [read_operation(oi) for oi in ops]
#    mats = rotate_mats(mats,avec)
    N,lims=8,1
    BZ = _gen_pts(N)#,lims)
    IBZ = op_reduce(mats,BZ)
    IBZ_pts = _d_to_arr(IBZ)
    
    _plt_pts(IBZ_pts,lims)
    
    IBZ_pts = rotate_points(IBZ_pts,avec)
    
    _plt_pts(IBZ_pts,lims)