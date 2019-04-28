#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 13:20:40 2019

@author: ryanday
"""

import numpy as np

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



def cell_edges(avec):
    '''
    Evaluate the edges forming an enclosing volume for the unit cell.
    
    *args*:
        - **avec**: numpy array of 3x3 float, lattice vectors
    
    *return*:
        - **edges**: numpy array of 24x2x3 float, indicating the endpoints of all
        bounding edges of the cell
    '''
    
    modvec = np.array([[np.mod(int(j/4),2),np.mod(int(j/2),2),np.mod(j,2)] for j in range(8)])
    edges = []
    for p1 in range(len(modvec)):
        for p2 in range(p1,len(modvec)):
            if np.linalg.norm(modvec[p1]-modvec[p2])==1:
                edges.append([np.dot(avec.T,modvec[p1]),np.dot(avec.T,modvec[p2])])
    edges = np.array(edges)
    return edges
    