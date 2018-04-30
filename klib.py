#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 18:28:22 2017

@author: ryanday
"""
'''
For now, given a set of pts--numpy array 3xN and a grain --# points between each of the N pts,
this will generate a Kobject, with labels, grain, and other attributes useful for plotting
In the future it would be nice to generate a klist based on an xcrysden calculation for example.
'''


import numpy as np

class kpath:
    
    def __init__(self,pts,grain=None,labels=None):
        ltype=type([])
        atype=type(np.array([]))
        if type(pts)==ltype:
            self.pts = pts
            self.grain = grain
            self.kpts = [] #will be the actual vector k points for calculation
            self.kcut = [] #will be a 1D array for plotting purposes
            self.kcut_brk = []
            self.ind_brk = [0]
            self.labels=labels
            self.points()
        elif type(pts)==atype:
            self.kpts = pts #if we are just feeding in an array of kpoints
    def points(self):
        #This method based on I.S. Elfimov function
        cutst = 0.0
        self.kcut_brk.append(cutst)
        self.kpts = [(self.pts[0])]
        self.kcut.append(cutst)
        for p in range(len(self.pts)-1):
            tempvec = np.array(self.pts[p+1])-np.array(self.pts[p])
            vecstep = tempvec/self.grain

            for i in range(1,self.grain+1):
                self.kpts.append(list(self.pts[p]+vecstep*float(i)))
                self.kcut.append(cutst+np.linalg.norm(vecstep)*float(i))
            cutst = self.kcut[-1]
            self.kcut_brk.append(cutst)
            self.ind_brk.append(len(self.kcut))
        self.kpts = np.array(self.kpts)
        return self.kpts
    
###ADD MESH GENERATION FOR e.g. DOS type-calculations    
    
def bvectors(a_vec):
    b_vec = 2*np.pi*np.array([(np.cross(a_vec[1],a_vec[2])/np.dot(a_vec[0],np.cross(a_vec[1],a_vec[2]))),(np.cross(a_vec[2],a_vec[0])/np.dot(a_vec[1],np.cross(a_vec[2],a_vec[0]))),(np.cross(a_vec[0],a_vec[1])/np.dot(a_vec[2],np.cross(a_vec[0],a_vec[1])))])
    return b_vec


def kmesh(ang,X,Y,kz):
    '''
    Take a mesh of kx and ky with fixed kz and generate a 3xN array of points
    '''
            
    kp = np.sqrt(X**2+Y**2)
    ph = np.arctan2(Y,X)
    if abs(ang)>0.0:
        X = kp*np.cos(ph-ang)
        Y = kp*np.sin(ph-ang)
    Zeff = kz*np.ones(np.shape(X))
    
    ph = np.reshape(ph,np.shape(X)[0]*np.shape(X)[1])
    k_arr = np.reshape(np.array([X,Y,Zeff]),(3,np.shape(X)[0]*np.shape(X)[1])).T
    return k_arr,ph