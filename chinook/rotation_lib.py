# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 20:04:24 2018


@author: rday

Various functions relevant to rotations

"""
import numpy as np
from math import atan2

    
def Euler(rotation):
    '''
    Euler rotation angle generation, Z-Y-Z convention, as defined for a user-defined
    rotation matrix. Special case for B = +/- Z*pi where conventional approach doesn't
    work due to division by zero, then Euler_A is zero and Euler_y is arctan(R10,R00)

    
    *args*:

        - **rotation**: numpy array of 3x3 float (rotation matrix)
        OR tuple/list of vector and angle (numpy array of 3 float, float) respectively
    
    *return*:

        - **Euler_A**, **Euler_B**, **Euler_y**: float, Euler angles associated with
        the given rotation.
        
    ***
    '''
    
    if type(rotation)==np.ndarray:
        rot_mat = rotation
    else:
        
        rot_mat = Rodrigues_Rmat(rotation[0],rotation[1])
    if abs(rot_mat[2,2])!=1.0:
    
    	Euler_A = atan2(rot_mat[1,2],rot_mat[0,2])
    
    	Euler_y = atan2(rot_mat[2,1],-rot_mat[2,0])
    
    	Euler_B = np.arccos(rot_mat[2,2])
    	if Euler_B>np.pi:
    		Euler_B = 2*np.pi- Euler_B
    
    	
    else:
        if np.sign(rot_mat[2,2])==1.0:
            Euler_B = 0.0
        else:
            Euler_B = np.pi
        if np.sign(rot_mat[2,2])==1:
            Euler_A = 0
            Euler_y = atan2(rot_mat[1,0],rot_mat[0,0])
        else:
            Euler_A = 0
            Euler_y = atan2(rot_mat[1,0],-rot_mat[0,0])

    return Euler_A,Euler_B,Euler_y



def Euler_to_R(Euler_A,Euler_B,Euler_y):
    
    '''
    Inverse of *Euler*, generate a rotation matrix from the Euler angles A,B,y
    with the same conventiona as in *Euler*.
   
    *args*:

        - **Euler_A**, **Euler_B**, **Euler_y**: float
    
    *return*:

        - numpy array of 3x3 float
        
    ***
    '''
    
    rot_y = Rodrigues_Rmat(np.array([0,0,1]),Euler_y)
    rot_B = Rodrigues_Rmat(np.array([0,1,0]),Euler_B)
    rot_A = Rodrigues_Rmat(np.array([0,0,1]),Euler_A)
    
    return np.dot(rot_A,np.dot(rot_B,rot_y))


    

def rotate_v1v2(v1,v2):
    
    '''
    This generates the rotation matrix for PRE-multiplication rotation:
    or written another way, defining R s.t. R.v1 = v2. This rotation will rotate
    the vector **v1** onto the vector **v2**.
    
    *args*:

        - **v1**: numpy array len 3 of float, input vector
        
        - **v2**: numpy array len 3 of float, vector to rotate into
        
    *return*:

        - **Rmat**: numpy array of 3x3 float, rotation matrix
    
    ***    
    '''
    
    v1 = v1/np.linalg.norm(v1)
    v2 = v2/np.linalg.norm(v2)
    vx=np.cross(v1,v2)
    cos=np.dot(v1,v2)
    vm=np.array([[0.0,-vx[2],vx[1]],[vx[2],0.0,-vx[0]],[-vx[1],vx[0],0.0]])
    if abs(cos+1)>1e-10:
        Rmat=np.identity(3)+vm+np.dot(vm,vm)/(1+cos)
    else:
        Rmat = np.identity(3)
        Rmat[2,2] = -1.0
    return Rmat


def Rodrigues_Rmat(nvec,theta):
    '''
    Following Rodrigues theorem for rotations, define a rotation matrix which
    corresponds to the rotation about a vector nvec by the angle theta, in radians.
    Works in pre-multiplication order (i.e. v' = R.v)
    
    *args*:

        - **nvec**: numpy array len 3 axis of rotation
        
        - **theta**: float radian angle of rotation counter clockwise for theta>0
        
    *return*:

        - **Rmat**: numpy array 3x3 of float rotation matrix
        
    ***    
    '''
    
    nvec = nvec/np.linalg.norm(nvec)
    Kmat = np.array([[0,-nvec[2],nvec[1]],[nvec[2],0,-nvec[0]],[-nvec[1],nvec[0],0]])
    Rmat = np.identity(3)+np.sin(theta)*Kmat+(1-np.cos(theta))*np.dot(Kmat,Kmat)
    return Rmat   


def rot_vector(Rmatrix):
    
    '''
    Inverse to *Rodrigues_Rmat*, take rotation matrix as input and return 
    the angle-axis convention rotations corresponding to this rotation matrix.
    
    *args*:

        - **Rmatrix**: numpy array of 3x3 float, rotation matrix
        
    *return*:
    
        - **nvec**: numpy array of 3 float, rotation axis
        
        - **theta**: float, rotation angle in float
        
    ***
    '''
    L,nvec=np.linalg.eig(Rmatrix)
    nvec = np.real(nvec[:,np.where(abs(L-1)<1e-10)[0][0]])
    theta = np.arccos((np.trace(Rmatrix)-1)/2)
    R_tmp = Rodrigues_Rmat(nvec,theta)
    if np.linalg.norm(R_tmp-Rmatrix)<1e-10:
        return nvec,theta
    else:
        R_tmp = Rodrigues_Rmat(nvec,-theta)
        if np.linalg.norm(R_tmp-Rmatrix)<1e-10:
            return nvec,-theta
        else:
            print('ERROR: COULD NOT DEFINE ROTATION MATRIX FOR SUGGESTED BASIS TRANSFORMATION!')
            return None