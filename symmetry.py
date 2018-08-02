#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 05:26:11 2018

@author: ryanday

Symmetry library for ubc_tbarpes

"""


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def _translate_(vt):
    '''
    Translate a vector, by another vector. 
    Length of vector v should be dim+1
    args: 
        vt numpy array numeric, length of v
    return:
        Translation matrix--numpy array of 4x4 float
    '''
    
    M = np.identity(4)
    M[:3,-1]=vt
    return M

def _rotate_(ax,ang):
    '''
    Rotate either a single vector, or N vectors about an axis vector by a fixed angle.
    args:
        ax rotation axis, numpy array of float, len(3)
        ang rotation angle, float
    return Rotation matrix np array 4x4 float
    '''
    ax = ax/np.linalg.norm(ax)
    ca = np.cos(ang)
    sa = np.sin(ang)
    xyc = ax[0]*ax[1]*(1-ca)
    xzc = ax[0]*ax[2]*(1-ca)
    yzc = ax[1]*ax[2]*(1-ca)
    R = np.array([[ca + ax[0]**2*(1-ca),xyc-ax[2]*sa,xzc+ax[1]*sa],
                  [xyc+ax[2]*sa,ca+ax[1]**2*(1-ca),yzc-ax[0]*sa],
                  [xzc-ax[1]*sa,yzc+ax[0]*sa,ca+ax[2]**2*(1-ca)]])
    Rt = np.identity(4)
    Rt[:3,:3] = R
    return Rt

def _reflect_(mirror):
    '''
    Reflect a vector, or array of vectors in a mirror plane--the component of the vector
    projected normal to the plane should be multiplied by -1, the in-plane projection unchanged
    args:
        
        mirror: normal direction of the mirror plane (numpy array of float len(3))
    return:
        reflection matrix numpy array of 4x4 float
    '''
    mirror = mirror/np.linalg.norm(mirror)
    M = np.array([mirror,mirror,mirror]).T*mirror
    M = (np.identity(3) - 2*M)
    Mt = np.identity(4)
    Mt[:3,:3] = M
    return Mt
    
def _invert_():
    '''
    Reflect all coordinate axes onto their negative self
    args: 
    return: inversion array 4x4 of numpy float
    '''
    return np.identity(4)*np.array([-1,-1,-1,1])

def _glide_(mirror,vt):
   '''
   Glide plane: reflection followed by translation
   args:
       mirror -- normal of mirror plane for reflection (numpy array len 3 of float)
       vt -- translation vector (numpy array len 3 of float)
   return: transformation matrix numpy array 4x4 of float
   '''
   return np.dot(_translate_(vt),_reflect_(mirror))

def _operate_(args,vecs):
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
    
    U = np.identity(4)
    if type(args[0])==tuple:
        for ui in args:
            tmp = _gen_mat_(ui)
            U = np.dot(tmp,U)
    else:
        U = _gen_mat_(args)
        
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
    
    
        
def _gen_mat_(args):
    ''' 
    Generate a symmetry operation
    The user sends a tuple organized with a single character to indicate the operation type:
        T -- translation, R -- rotation, M -- mirror, I -- inversion, G -- glide
    The other elements of the tuple are the corresponding arguments, e.g. for glide, these will
    be the mirror plane and translation vector, ('G',np.array([0,0,1]),np.array([0.5,0.5,0.0]))
    Composite operations can also be defined, in which case we iterate over each tuple in a list of
    '''
    
    if args[0]=='T':
        M = _translate_(args[1])
    elif args[0] == 'R':
        M = _rotate_(*args[1:])
    elif args[0] == 'M':
        M = _reflect_(args[1])
    elif args[0] == 'I':
        M = _invert_()
    elif args[0] == 'G':
        M = _glide_(*args[1:])
    return M

def _seed_rnd(N):
    '''
    Generate N random 3D vectors
    '''
    pts = np.array([[-1+2*np.random.random(),-1+2*np.random.random(),-1+2*np.random.random()] for ni in range(N)])
    return pts
    

def _plt_pts_(pts,pts2):
    fig = plt.figure()
    ax= fig.add_subplot(111,projection='3d')
    ax.scatter(pts[:,0],pts[:,1],pts[:,2],c='r')
    ax.scatter(pts2[:,0],pts2[:,1],pts2[:,2],c='b')



if __name__=="__main__":
    

    b = _seed_rnd(40)
    
    br = _operate_((('M',np.array([1,0,0]))),b)
    
    _plt_pts_(b,br)