# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 20:04:24 2018


@author: rday

Various functions relevant to rotations

"""
import numpy as np

def Euler(rotation):    
#def Euler(n,t):
    '''
    Extract the Euler angles for a ZYZ rotation around n by t
    args:
        rotation --> numpy array of 3x3 float (rotation matrix) OR tuple/list of vector and angle (numpy array of 3 float, float) respectively
        #n --> np.array x3 float for axis
        #t --> float radian angle of rotation counter clockwise for t>0
    Has special case for B = +/- Z*pi where conventional approach doesn't work due to division by zero
    '''
    if type(rotation)==np.ndarray:
        R = rotation
    else:
        
        R = Rodrigues_Rmat(rotation[0],rotation[1])

    b = np.arccos(R[2,2])
    sb = np.sin(b)
    a,y=0.0,0.0
    if abs(sb)>10.0**-6:
        y = np.arctan2(R[1,2],-R[0,2])
        a = np.arctan2(R[2,1],R[2,0])
    else: #alternate definition using arcos and arcsin, 1/10/2018
        a = 0.0#0.5*(np.arccos(R[1,1]) + np.arcsin(R[1,0]))
        if R[2,2]>0:
            y = np.arctan2(R[0,1],R[1,1])# - np.arccos(R[1,1]))
        else:
            y = np.arctan2(R[0,1],R[1,1])
            
#        a=[np.arctan2(R[1,0],R[0,0]),np.arctan2(R[1,0],-R[0,0])]
#
#        if R[2,2]<0:
#            a = np.arctan2(R[1,0],R[0,0])
#
#        else:
#            a = np.arctan2(R[1,0],-R[0,0])
#    print('A: {:0.04f},B:{:0.04f},y:{:0.04f}'.format(a,b,y))
    return a,b,y


def Rodrigues_Rmat(n,t):
    '''
    Rodrigues theorem for rotations. 
    args:
        n --> np.array x 3 axis of rotation
        t --> float radian angle of rotation counter clockwise for t>0
    return:
        R --> np.array (3x3) of float rotation matrix
    '''
    n = n/np.linalg.norm(n)
    K = np.array([[0,-n[2],n[1]],[n[2],0,-n[0]],[-n[1],n[0],0]])
    R = np.identity(3)+np.sin(t)*K+(1-np.cos(t))*np.dot(K,K)
    return R   


def rot_vector(Rmatrix):
    '''
    Take rotation matrix as input and return the angle-axis convention rotations corresponding to this rotation matrix
    '''
    L,u=np.linalg.eig(Rmatrix)
    uv = np.real(u[:,np.where(abs(L-1)<1e-10)[0][0]])
    th = np.arccos((np.trace(Rmatrix)-1)/2)
    R_tmp = Rodrigues_Rmat(uv,th)
    if np.linalg.norm(R_tmp-Rmatrix)<1e-10:
        return uv,th
    else:
        R_tmp = Rodrigues_Rmat(uv,-th)
        if np.linalg.norm(R_tmp-Rmatrix)<1e-10:
            return uv,-th
        else:
            print('ERROR: COULD NOT DEFINE ROTATION MATRIX FOR SUGGESTED BASIS TRANSFORMATION!')
            return None