# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 16:27:40 2018

@author: rday
"""

import numpy as np
import matplotlib.pyplot as plt
import ubc_tbarpes.klib as klib

def not_point(point):
    return np.mod(point+1,2)

def neighbours(point):
    tmp = np.array([point,point,point])
    return np.mod(tmp+np.identity(3),2)

def corners():
    '''
    Establish the shortest main diagonal of a cube of points, so as to establish
    the main diagonal for tetrahedral partitioning of the cube
    return:
        main -- tuple of 2 integers indicating the cube coordinates
        cube -- numpy array of 8 corners (8x3) float
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
    return:
        tetra -- numpy array of 6 x 4 x 3 int, indicating the corners of the 6 tetrahedra
    '''
    main,cube = corners()
    tetra = np.zeros((6,4,3))
    tetra[:,:2,:] = cube[main]
    c2 = neighbours(main[0])
    for i in range(3):
        c2i = c2[i]
        c3_options = neighbours(c2i)
        c3f = c3_options[np.where(np.linalg.norm(c3_options-main[0],axis=1)>0)]
        tetra[2*i:2*(i+1),2,:] = np.array([c2i,c2i])
        tetra[2*i:2*(i+1),3,:] = c3f
        
    return tetra

def tet_inds():
    '''
    Generate, for a single cube, the tetrahedral designations, for the following conventional numbering:
         6 o ---- o 7       
         /      / |
      4 o ---- o5 o 3
        |      | /
      0 o ---- o 1
      
      with 2 hidden from view. 
      Defining the real-index spacing between adjacent cubes in a larger array, we can apply this simple prescription
      to define the spanning tetrahedra over the larger k-mesh
    '''
    corn_dict = {'000':0,'100':1,'010':2,'110':3,'001':4,'101':5,'011':6,'111':7}
    
    tetra_vec = tetrahedra().astype(int)
    tetra_inds = np.array([[corn_dict['{:d}{:d}{:d}'.format(*ti)] for ti in tetra_vec[j]] for j in range(6)])
    return tetra_inds

def mesh_tetra(avec,N):
    '''
    Generate a mesh of points spanning several lattice points in reciprocal space: bz_mesh_raw
    From this mesh, reduce to simply a set of points which define the first Brillouin zone.
    On this mesh, define a set of identical tetrahedra which can be used for the sake of interpolation of
    quantities defined over the Brillouin zone. The tetrahedra are defined first in terms of the lattice space
    mesh indices, and then transformed to the BZ mesh indices. The BZ mesh indices can then be passed as a Kobj
    to TB and diagonalized, from which e.g. DOS can be computed. 
    args:
        avec -- lattice vectors numpy array of 3x3 float
        N -- integer, or tuple/list of 3 integers for defining the k-mesh density
    return:
        raw_mesh[bz] -- numpy array of k-points (Mx3 float) for diagonalization over the BZ spanning set of k-points
        mesh_tet -- numpy array of tetrahedra spanning the same region of k-space, shape Kx4 with each element an integer
        indicating the coordinates in raw_mesh[bz] of the tetrahedra corner. Each set of 6 consecutive rows compose the corners
        of a cube in k-space
    
    '''
    b_vec = klib.bvectors(avec)
    rlpts = np.dot(klib.region(1),b_vec)    
    raw_mesh = klib.raw_mesh(rlpts,N)
    bz = klib.mesh_reduce(rlpts,raw_mesh,inds=True)
    points = raw_mesh[bz]
    Nx = len(set(points[:,0]))
    Ny = len(set(points[:,1]))
    ti = tet_inds()
    
    mesh_tet = []
    for ii in range(len(bz)):
        try:
            mesh_tet.append(bz[propagate(ii,Nx,Nx*Ny)[ti]])
        except IndexError:
            continue
    mesh_tet = np.array(mesh_tet)
    
    shp = np.shape(mesh_tet)
    try:
        mesh_tet = mesh_tet.reshape((shp[0]*shp[1],shp[2]))
        for i in range(len(bz)):
            mesh_tet[np.where(mesh_tet==bz[i])]=i    
        return raw_mesh[bz],mesh_tet
#   
    except IndexError:
        print('WARNING: NO K-POINTS FOUND, CHOOSE A FINER K-POINT GRID')
        return None,None
#   


def propagate(i,Nr,Nc):
    return i + np.array([0,1,Nr,Nr+1,Nc,Nc+1,Nr+Nc,Nr+Nc+1])
    



if __name__ == "__main__":
    
#    a = tetrahedra()
#    ti = tet_inds()
    avec = np.array([[5,0,0],[0,5,0],[0,0,5]])
#    bz = klib.b_zone(avec,4)
    
    bz,mtet = mesh_tetra(avec,8)
    