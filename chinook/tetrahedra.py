# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 16:27:40 2018

@author: rday
"""

import numpy as np
import matplotlib.pyplot as plt
import chinook.klib as klib

def not_point(point):
    '''
    
    Inverse of point, defined in an N-dimensional binary coordinate frame
    
    *args*:
        
        - **point**: int or numpy array of int between 0 and 1
    
    *return*:
        
        - numpy array of int, NOT gate applied to the binary vector point

    ***
    '''
    return np.mod(point+1,2)

def neighbours(point):
    '''
    
    For an unit cube, we can define the set of 3 nearest neighbours by performing
    the requisite modular sum along one of the three Cartesian axes. In this way,
    for an input point, we can extract its neighbours easily.
    
    *args*:
        
        - **point**: numpy array of 3 int, all either 0 or 1
    
    *return*:
        
        - numpy array of 3x3 int, indicating the neighbours of **point** on the
        unit cube.
        
    ***    
    '''
    tmp = np.array([point,point,point])
    return np.mod(tmp+np.identity(3),2)

def corners():
    '''
    Establish the shortest main diagonal of a cube of points, so as to establish
    the main diagonal for tetrahedral partitioning of the cube
    
    *return*:
        
        **main**: tuple of 2 integers indicating the cube coordinates
        
        **cube**: numpy array of 8 corners (8x3) float
    
    ***
    '''
    cube = np.array([[i,j,k] for k in range(2) for j in range(2) for i in range(2)])
    diagonals = np.array([cube[7-ii]-cube[ii] for ii in range(4)])
    min_ind = np.where(np.linalg.norm(diagonals)==np.linalg.norm(diagonals).min())[0][0]
    main = sorted([min_ind,7-min_ind])
    return main,cube
    
def tetrahedra():
    
    '''
    
    Perform partitioning of a cube into tetrahedra. The indices can then be
    dotted with some basis vector set to put them into the proper coordinate frame.
    
    *return*:
        
        - **tetra**: numpy array of 6 x 4 x 3 int, indicating the corners
        of the 6 tetrahedra
        
    ***
    '''
    main,cube = corners()
    tetra = np.zeros((6,4,3))
    tetra[:,:2,:] = cube[main]
    origin_neighbours = neighbours(main[0])
    for i in range(3):
        origin_neighbours_i = origin_neighbours[i]
        neighbour_options = neighbours(origin_neighbours_i)
        choices = neighbour_options[np.where(np.linalg.norm(neighbour_options-main[0],axis=1)>0)]
        tetra[2*i:2*(i+1),2,:] = np.array([origin_neighbours_i,origin_neighbours_i])
        tetra[2*i:2*(i+1),3,:] = choices
        
    return tetra

def tet_inds():
    '''
    Generate, for a single cube, the tetrahedral designations, 
    for the following conventional numbering:
         6 o ---- o 7       
         /      / |
      4 o ---- o5 o 3
        |      | /
      0 o ---- o 1
      
      with 2 hidden from view (below 6, and behind the line-segment connecting 4-5). 
      Here drawn with x along horizontal, z into plane, y vertical
      Defining the real-index spacing between adjacent cubes in a larger array, we can apply this simple prescription
      to define the spanning tetrahedra over the larger k-mesh
    
    *return*:
        
        - **tetra_inds**: numpy array of integer (6x4), with each
        row containing the index of the 4 tetrahedral vertices. Together, for
        of a set of neighbouring points on a grid, we divide into a set of covering
        tetrahedra which span the volume of the cube.
        
    ***
    '''
#    corn_dict = {'000':0,'100':1,'010':2,'110':3,'001':4,'101':5,'011':6,'111':7}
    corn_dict = {'000':0,'001':1,'100':2,'101':3,'010':4,'011':5,'110':6,'111':7}
    
    tetra_vec = tetrahedra().astype(int)
    tetra_inds = np.array([[corn_dict['{:d}{:d}{:d}'.format(*ti)] for ti in tetra_vec[j]] for j in range(6)])
    return tetra_inds


def gen_mesh(avec,N):
    '''
    Generate a mesh of points in 3-dimensional momentum space over the first
    Brillouin zone. These are defined first in terms of recirocal lattice vectors,
    
    i.e. from 0->1 along each, and then are multiplied by the rec. latt. vectors 
    themselves. Note that this implicitly provides a mesh which is not centred
    at zero, but has an origin at the rec. latt. vector (0,0,0)
    
    *args*:
        
        - **avec**: numpy array of 3x3 float, lattice vectors

        - **N**: int, or tuple of 3 int, indicating the number of points along
        each of the reciprocal lattice vectors        
        
    ***
    '''
    
    if type(N)==int:
        N = (N,N,N)
        
    b_vec = klib.bvectors(avec)
    x,y,z = np.linspace(0,1,N[0]),np.linspace(0,1,N[1]),np.linspace(0,1,N[2])
    X,Y,Z = np.meshgrid(x,y,z)
    X,Y,Z = X.flatten(),Y.flatten(),Z.flatten()
    pts = np.dot(np.array([[X[i],Y[i],Z[i]] for i in range(len(X))]),b_vec)

    return pts

def mesh_tetra(avec,N):
    '''
    An equivalent definition of a spanning grid over the Brillouin zone is just
    one which spans the reciprocal cell unit cell. Translational symmetry imposes
    that this partitioning is equivalent to the conventional definition of the 
    Brillouin zone, with the very big advantage that we can define a rectilinear
    grid which spans this volume in a way which can not be done for most
    Bravais lattices in R3. 
    
    *args*:
        
        - **avec**: numpy array of 3x3 float, lattice vectors
        
        - **N**: int, or iterable of 3 int which define the density of the mesh
        over the Brillouin zone.
        
    *return*:
        
        - **pts**: numpy array of Mx3 float, indicating the points in momentum space
        at the vertices of the mesh
        
        - **mesh_tet**: numpy array of Lx4 int, indicating the L-tetrahedra
        which partition the grid
      
    ***
    '''
    
    if type(N)==int:
        N = (N,N,N)
    pts = gen_mesh(avec,N)

    ti = tet_inds()
    
    mesh_tet = []
    for ii in range(len(pts)):
        test_tetra = propagate(ii,N[2],N[0])[ti]
        for t in test_tetra:
            try:
                if t.max()>=len(pts) or t.max()<0:
                    continue
                else:
                    mesh_tet.append(t)
            except KeyError:
                continue
         
    mesh_tet = np.array(mesh_tet)
    
    if len(mesh_tet)>0:
        return pts,mesh_tet
    else:
        print('WARNING: NO K-POINTS FOUND, CHOOSE A FINER K-POINT GRID')
        return None,None
       

def propagate(i,Nr,Nc):
    
    '''
    Distribute the generic corner numbering convention defined for a cube at the 
    origin to a cube starting at some arbitrary point in our grid. Excludes the
    edge points as starting points, so that all cubes are within the grid.
    
    *args*:
        
        - **i**: int, index of origin
        
        - **Nr**: int, number of rows in grid
        
        - **Nc**: int, number of columns in grid
        
    *return*:
        
        - **numpy array of int, len 8 corresponding to the re-numbering of the
        corners of the cube.
    
    ***
    '''
    if (np.mod(i,Nr)<(Nr-1)) and (np.mod(Nr+i,Nr*Nc)>np.mod(i,Nr*Nc)):
        
        return i + np.array([0,1,Nr,Nr+1,Nr*Nc,Nr*Nc+1,Nr*(1+Nc),Nr*(1+Nc)+1])
    else:
        return -1*np.ones(8)
