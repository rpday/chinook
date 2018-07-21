# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 11:21:25 2018

@author: rday
"""

import numpy as np
import matplotlib.pyplot as plt
import v3_def as v3




def GCD(a,b):
    '''
    Basic greatest common denominator function. First find all divisors of each a and b.
    Then find the maximal common element of their divisors.
    args: a,b int
    return: GCD of a,b 
    '''
    dA,dB = divisors(a),divisors(b)
    return max(dA[i] for i in range(len(dA)) if dA[i] in dB)
    
def LCM(a,b):
    '''
    Basic lowest-common multiplier for two values a,b. Based on idea that LCM is just the
    product of the two input, divided by their greatest common denominator.
    args: a,b, int
    return: LCM
    '''
    return int(a*b/GCD(a,b))

def divisors(a):
    '''
    Iterate through all integer divisors of integer input
    args: a int
    return list of divisors
    '''
    return [int(a/i) for i in range(1,a+1) if (a/i)%1==0]




def LCM_3(a,b,c):
    '''
    For generating spanning vectors, require lowest common multiple of 3 integers, itself just
    the LCM of one of the numbers, and the LCM of the other two. 
    args: a,b,c int
    return LCM of the three numbers
    '''
    
    return LCM(a,LCM(b,c))

def iszero(a):
    
    return np.where(a==0)[0]

def nonzero(a):
    
    return np.where(a!=0)[0]

def abs_to_frac(avec,vec):
    '''
    Quick function for taking a row-ordered matrix of lattice vectors: 
        | a_11  a_12  a_13  |
        | a_21  a_22  a_23  |
        | a_31  a_32  a_33  |
    and using to transform a vector, written in absolute units, to fractional units.
    Note this function can be used to broadcast over N vectors you would like to transform
    args: avec -- numpy array of lattice vectors, ordered by rows (numpy array 3x3 of float)
          vec -- numpy array of 3 float (or Nx3) -- basis vector to be transformed to fractional coordinates
     return: re-defined basis vector (len 3 array of float), or Nx3 array of float
    
    '''
    return np.dot(vec,np.linalg.inv(avec))

def frac_to_abs(avec,vec):
    '''
    Same as abs_to_frac, but in opposite direction--from fractional to absolute coordinates
    args:
        avec -- numpy array of lattice vectors, row-ordered (3x3 numpy array of float)
        vec -- numpy array of input vector(s) as Nx3 float
    return: vec in units of absolute coordinates (Angstrom), N x 3 array of float
        
    '''
    return np.dot(vec,avec)


def p_vecs(miller,avec):
    '''
    Produce the vectors p, as defined by Ceder, to be used in defining spanning vectors for plane normal
    to the Miller axis
    args: miller numpy array of len 3 float
        avec numpy array of size 3x3 of float
    return p, numpy array size 3x3 of float
    '''
    n_zero = nonzero(miller)
    zero = iszero(miller)
    p = np.zeros((3,3))
    if len(n_zero)==3:
        M = LCM_3(*miller)
        p = np.array([avec[0]*M/miller[0],
                      avec[1]*M/miller[1],
                      avec[2]*M/miller[2]])
    elif len(n_zero)==2:
        M = LCM(*miller[n_zero])
        p[n_zero[0]] = M/miller[n_zero[0]]*avec[n_zero[0]]
        p[n_zero[1]] = M/miller[n_zero[1]]*avec[n_zero[1]]
        p[zero[0]] = p[n_zero[0]] + avec[zero[0]]
    elif len(n_zero)==1:
        p[n_zero[0]] = np.zeros(3)
        p[zero[0]] = avec[zero[0]]
        p[zero[1]] = avec[zero[1]]
        
    return p


def v_vecs(miller,avec):
    '''
    Wrapper for functions used to determine the vectors used to define the new,
    surface unit cell.
    args: 
        miller -- numpy array of int len(3) Miller indices for surface normal
        avec -- numpy array of 3x3 float -- Lattice vectors
    return:
        new surface unit cell vectors numpy array of 3x3 float
    '''
    p = p_vecs(miller,avec)
    v = np.zeros((3,3))
    v[0] = p[1]-p[0]
    v[1] = p[2]-p[0]
    v[2] = v3.find_v3(v[0],v[1],avec,40)
    return v

def basal_plane(v):
    '''
    Everything is most convenient if we redefine the basal plane of the surface normal to be oriented within a 
    Cartesian plane. To do so, we take the v-vectors. We get the norm of v1,v2 and then find the cross product with
    the z-axis, as well as the angle between these two vectors. We can then rotate the surface normal onto the z-axis.
    In this way we conveniently re-orient the v1,v2 axes into the Cartesian x,y plane.
    ARGS:
        v -- numpy array 3x3 float
    return:
        vp -- rotated v vectors, numpy array 3x3 of float
        R -- rotation matrix to send original coordinate frame into the rotated coordinates.
    '''
    norm = np.cross(v[0],v[1])
    norm = norm/np.linalg.norm(norm)
    x = np.cross(norm,np.array([0,0,1]))
    sin = np.linalg.norm(x)
    x = x/sin
    cos = np.dot(norm,np.array([0,0,1]))
    u = np.array([[0,-x[2],x[1]],[x[2],0,-x[0]],[-x[1],x[0],0]])
    uu = np.array([x[0]*x,x[1]*x,x[2]*x])
    R = cos*np.identity(3) + sin*u + (1-cos)*uu
    
    vp = np.around(np.dot(v,R.T),4)
    
    return vp,R


def par(avec):
    '''
    Definition of the parallelepiped, as well as a containing region within the 
    Cartesian projection of this form which can then be used to guarantee correct
    definition of the new cell basis. The parallelipiped is generated, and then
    its extremal coordinates established, from which a containing parallelepiped is
    then defined. The parallelepiped (blue) is plotted along with a semi-transparent
    (red) container box
    args:
        avec -- numpy array of 3x3 float
    return:
        vert -- numpy array  8x3 float vertices of parallelepiped
        box_pts -- numpy array 8 x 3 float vertices of containing box
    '''
    
    pts = np.array([[int(i/4),int(np.mod(i/2,2)),int(np.mod(i,2))] for i in range(8)])
    vert = np.dot(pts,avec)
    alpha,omega = np.array([vert[:,0].min(),vert[:,1].min(),vert[:,2].min()]),np.array([vert[:,0].max(),vert[:,1].max(),vert[:,2].max()])
    box = np.identity(3)*(omega-alpha)
    box_pts = np.dot(pts,box)
    box_pts = np.array([c+alpha for c in box_pts])
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(vert[:,0],vert[:,1],vert[:,2])
    ax.scatter(box_pts[:,0],box_pts[:,1],box_pts[:,2],c='r',alpha=0.3)
    return vert,box_pts


def gen_cov_mat(avec,vvec):
    
    C_33 = vvec*np.linalg.inv(avec)
    C_44 = np.zeros((4,4))
    C_44[:3,:3] = C_33
    C_44[3,3] = 1
    
    return C_44

if __name__=="__main__":
    
    avec = np.identity(3)
    miller =np.array([0,1,0])
    avec = np.array([[0,4.736235,0],[4.101698,-2.368118,0],[0,0,13.492372]])
    miller = np.array([1,0,4])
    
    vn = v_vecs(miller,avec)
    
    vn_b,R = basal_plane(vn)
#    pipe,box = par(vn)
    
    C = gen_cov_mat(avec,vn)
    
    C2 = gen_cov_mat(avec,vn_b)
    




    