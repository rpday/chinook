# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 20:04:24 2018


@author: rday

Various functions relevant to rotations

"""
import numpy as np
from math import atan2

#def Euler(rotation):    
##def Euler(n,t):
#    '''
#    Extract the Euler angles for a ZYZ rotation around n by t
#    args:
#        rotation --> numpy array of 3x3 float (rotation matrix) OR tuple/list of vector and angle (numpy array of 3 float, float) respectively
#        #n --> np.array x3 float for axis
#        #t --> float radian angle of rotation counter clockwise for t>0
#    Has special case for B = +/- Z*pi where conventional approach doesn't work due to division by zero
#    '''
#    if type(rotation)==np.ndarray:
#        R = rotation
#    else:
#        
#        R = Rodrigues_Rmat(rotation[0],rotation[1])
#    B = [np.arccos(R[2,2]),-np.arccos(R[2,2])]
#    if abs(R[2,2])!=1.0:    
#        a= [np.arctan2(R[2,1]/np.sin(B[0]),-R[2,0]/np.sin(B[0])),atan2(R[2,1]/np.sin(B[1]),-R[2,0]/np.sin(B[1]))]
#        y= [atan2(R[1,2]/np.sin(B[0]),R[0,2]/np.sin(B[0])),atan2(R[1,2]/np.sin(B[1]),R[0,2]/np.sin(B[1]))]
#    elif abs(R[2,2])==1.0:
#            a=[atan2(R[1,0],R[0,0]),atan2(R[1,0],-R[0,0])]
#            y = [0.0,0.0]
#    if (a[0]**2+B[0]**2+y[0]**2)<(a[1]**2+B[1]**2+y[1]**2):
#        flag = 0
#    else:
#        flag = 1
#    return a[flag],B[flag],y[flag]


#    b = np.arccos(R[2,2])
#    sb = np.sin(b)
#    a,y=0.0,0.0
#    if abs(sb)>10.0**-6:
#        y = atan2(R[1,2],-R[0,2])
#        a = atan2(R[2,1],R[2,0])
##        y = np.arctan2(R[1,2],-R[0,2])
##        a = np.arctan2(R[2,1],R[2,0])
#    else: #alternate definition using arcos and arcsin, 1/10/2018
#        a = 0.0#0.5*(np.arccos(R[1,1]) + np.arcsin(R[1,0]))
#        if R[2,2]>0:
#            y = atan2(R[0,1],R[1,1])
##            y = np.arctan2(R[0,1],R[1,1])# - np.arccos(R[1,1]))
#        else:
#            y = atan2(R[0,1],R[1,1])
#            y = np.arctan2(R[0,1],R[1,1])
            
#        a=[np.arctan2(R[1,0],R[0,0]),np.arctan2(R[1,0],-R[0,0])]
#
#        if R[2,2]<0:
#            a = np.arctan2(R[1,0],R[0,0])
#
#        else:
#            a = np.arctan2(R[1,0],-R[0,0])
#    print('A: {:0.04f},B:{:0.04f},y:{:0.04f}'.format(a,b,y))
#    return a,b,y
    
def Euler(rotation):
    '''
    Euler rotation angle generation, updated 17/10/2018 (cf Labbook for details)
    args:
        rotation --> numpy array of 3x3 float (rotation matrix) OR tuple/list of vector and angle (numpy array of 3 float, float) respectively
    return 3 numbers: A,B,y
    Has special case for B = +/- Z*pi where conventional approach doesn't work due to division by zero
    '''
    
    if type(rotation)==np.ndarray:
        R = rotation
    else:
        
        R = Rodrigues_Rmat(rotation[0],rotation[1])
    if abs(R[2,2])!=1.0:
    
    	A = atan2(R[1,2],R[0,2])
    
    	y = atan2(R[2,1],-R[2,0])
    
    	B = np.arccos(R[2,2])
    	if B>np.pi:#abs(np.sin(B)*np.cos(A)-R[0,2]) >1e-10:
    		B = 2*np.pi- B
    
    	
    else:
        if np.sign(R[2,2])==1.0:
            B = 0.0
        else:
            B = np.pi
        if np.sign(R[2,2])==1:
            A = 0
            y = atan2(R[1,0],R[0,0])
        else:
            A = 0
            y = atan2(R[1,0],-R[0,0])
#    Rtmp = Euler_to_R(A,B,y)
#    if np.linalg.norm(Rtmp-R)<1e-3:
#        print('Valid Euler angles')
    return A,B,y
#    else:
#        print('Error in Euler computation')
#        print(A,B,y)
#        return 0,0,0


def Euler_to_R(A,B,y):
    
    Ry = Rodrigues_Rmat(np.array([0,0,1]),y)
    RB = Rodrigues_Rmat(np.array([0,1,0]),B)
    RA = Rodrigues_Rmat(np.array([0,0,1]),A)
    
    return np.dot(RA,np.dot(RB,Ry))


    

def rotate_v1v2(v1,v2):
    '''
    This generates the rotation matrix for PRE-multiplication rotation: i.e. R.v1 = v2
        
    '''
    
    v1 = v1/np.linalg.norm(v1)
    v2 = v2/np.linalg.norm(v2)
    vx=np.cross(v1,v2)
    c=np.dot(v1,v2)
    vm=np.array([[0.0,-vx[2],vx[1]],[vx[2],0.0,-vx[0]],[-vx[1],vx[0],0.0]])
    if abs(c+1)>1e-10:
        R=np.identity(3)+vm+np.dot(vm,vm)/(1+c)
    else:
        R = np.identity(3)
        R[2,2] = -1.0
    return R


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