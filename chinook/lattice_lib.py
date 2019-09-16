#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 13:20:40 2019

@author: ryanday
"""

import numpy as np
import matplotlib.pyplot as plt


class lattice:

    '''
    Primarily a utility class, for use in visualizing lattice and
    basis. Also includes some tools for converting between different
    unit conventions.

    '''
    
    def __init__(self,avec,basis):
        
        self.avec = avec
        self.ivec = np.linalg.inv(avec)
        self.pos,self.fpos = self.parse_basis(basis,3)
        self.edges = self.cell_edges()
        
        
    def parse_basis(self,basis,nmax):
        '''
        Take orbital basis and establish all equivalent locations
        of each atom in the basis within a region near the origin

        *args*:

            - **basis**: list of orbital objects

            - **nmax**: int, number of neighbouring cells to consider

        *return*:

            - **atoms**: dictionary, key values indications atoms, all positions

            - **frac_posns**: same as **atoms**, but in lattice vector units

        ***
        '''
        
        atoms = {}
        frac_posns = {}

        for bi in basis:
            if bi.atom not in atoms:
                atoms[bi.atom] = [tuple(bi.pos)]
            else:
                atoms[bi.atom].append(tuple(bi.pos))
        for ai in atoms:
            atoms[ai] = np.array(list(dict.fromkeys(atoms[ai])))
                     
            fpos =self.frac_pos(atoms[ai])
            frac_posns[ai] = neighbours(fpos,nmax)
            atoms[ai] = np.einsum('ij,ki->kj',self.avec,frac_posns[ai])
            
        return atoms,frac_posns
            
        
        
    def frac_pos(self,posns):
        '''
        Inverse multiplication of lattice vectors with position vector, 
        to get position in units of lattice vectors, rather than direct units
        of Angstrom

        *args*:

            - **posns**: numpy array of Nx3 float

        *return*:

            - numpy array of Nx3 float

        ***
        '''
        
        return np.einsum('ji,kj->ki',self.ivec,posns)
        
        
    def draw_lattice(self,ax=None):
        '''
        Plotting utility function, display unit cell parallelepiped, and
        atoms inside

        *kwargs*:
            
            - **ax**: matplotlib Axes, for plotting onto existing axes

        *return*:

            - **ax**: matplotlib Axes, for further editing of plots

        ***
        '''
        
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111,projection='3d')
        
        for ii in range(len(self.edges)):
            ax.plot(self.edges[ii,:,0],self.edges[ii,:,1],self.edges[ii,:,2],c='k')
        
        for ai in self.fpos:
            ax.scatter(self.pos[ai][:,0],self.pos[ai][:,1],self.pos[ai][:,2])
            
        return ax
            
    def cell_edges(self):

        '''
        Evaluate the edges forming an enclosing volume for the unit cell.
        
        *args*:

            - **avec**: numpy array of 3x3 float, lattice vectors
        
        *return*:

            - **edges**: numpy array of 24x2x3 float, indicating the endpoints of all
            bounding edges of the cell

        ***
        '''
        
        modvec = np.array([[np.mod(int(j/4),2),np.mod(int(j/2),2),np.mod(j,2)] for j in range(8)])
        edges = []
        for p1 in range(len(modvec)):
            for p2 in range(p1,len(modvec)):
                if np.linalg.norm(modvec[p1]-modvec[p2])==1:
                    edges.append([np.dot(self.avec.T,modvec[p1]),np.dot(self.avec.T,modvec[p2])])
        edges = np.array(edges)
        return edges
    
    

def lattice_pars_to_vecs(norm_a,norm_b,norm_c,ang_a,ang_B,ang_y):
    
    '''

    A fairly standard way to define lattice vectors is in terms of the vector lengths,
    and their relative angles--defining in this way a parallelepiped unit cell. Use
    this notation to translate into a numpy array of 3x3 array of float
    
    *args*:

        - **norm_a**, **norm_b**, **norm_c**: float, length vectors, Angstrom
        
        - **ang_a**, **ang_B**, **ang_y**: float, angles between (b,c), (a,c), (a,b) in degrees
    
    *return*:

        - numpy array of 3x3 float: unit cell vectors
    
    ***
    '''
    
    rad = np.pi/180
    ang_a *=rad
    ang_B *=rad
    ang_y *=rad
    
    vec1 = np.array([norm_a,0,0])
    vec2 = norm_b*np.array([np.cos(ang_y),np.sin(ang_y),0])
    vec3 = norm_c*np.array([np.cos(ang_B),(np.cos(ang_a)-np.cos(ang_B)*np.cos(ang_y))/np.sin(ang_y),0])
    vec3[2] = np.sqrt(norm_c**2 - vec3[0]**2 - vec3[1]**2)
    
    return np.around(np.array([vec1,vec2,vec3]),5)



def neighbours(pos,num):
    '''

    Build series of lattice points using the pos arrays, out to some fixed number of points away
    from origin

    *args*:

        - **pos**: numpy array of 3x3 float

        - **num**: int, number of sites to consider

    *return*:

        - **inside**: numpy array of Nx3 float, all points in neighbouring region of lattice
    
    num_symm= 2*num+1
    
    points = np.array([[int(i/num_symm**2)-num,int(i/num_symm)%num_symm-num,i%num_symm-num] for i in range((num_symm)**3)])
    
    inside = []
    for pi in pos:
        all_points = points + pi
        all_points = all_points[np.where((all_points[:,0]>=0) & (all_points[:,0]<1) & (all_points[:,1]>=0) & (all_points[:,1]<1) & (all_points[:,2]>=0) & (all_points[:,2]<1))]
        inside.append(*all_points)
        
    return np.array(inside)
    



    