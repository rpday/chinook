# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 14:20:31 2018

@author: rday
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
#from matplotlib.colors import ListedColormap,BoundaryNorm
import matplotlib.cm as cm
import PtCoO2
import ubc_tbarpes.operator_library as operlib




def find_interceptions(TB,indices,kmax,Nt,Nk,E):
    theta = np.linspace(0,2*np.pi,Nt)
    k = np.linspace(0,kmax,Nk)
    points = []
    for t in theta:
        kpts = np.array([k*np.cos(t),k*np.sin(t),np.zeros(len(k))]).T
        TB.Kobj.kpts = kpts
        TB.solve_H()
        tmp_points = []
        for bi in indices:
            
            kn = np.where(abs(TB.Eband[:,bi]-E)==abs(TB.Eband[:,bi]-E).min())[0][0]
            side_E = np.sign(TB.Eband[kn,bi]-E)
            side_dE = (np.sign(TB.Eband[np.mod(kn-1,len(kpts)),bi]-E),np.sign(TB.Eband[np.mod(kn+1,len(kpts)),bi]-E))
            if side_dE[0]!=side_E:
                pts = (np.mod(kn-1,len(kpts)),kn)
            else:
                pts = (kn,np.mod(kn+1,len(kpts)))
            
            
            k_inter = (kpts[pts[1]]-kpts[pts[0]])/(TB.Eband[pts[1],bi]-TB.Eband[pts[0],bi])*(E-TB.Eband[pts[0],bi])+kpts[pts[0]]
            tmp_points.append(k_inter)
        points.append(tmp_points)
    return np.array(points)


def plot_fs(points):
    
    fig = plt.figure()
    for bi in np.shape(points)[1]:
        plt.plot(points[:,bi,:])
        
def operator(O,TB,inds,points,vlims=(-1,1)):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    Ovalues = np.zeros(np.shape(points)[:2])
    cmappable = 0 
    p = []
    for bi in range(np.shape(points)[1]):
        
        TB.Kobj.kpts = points[:,bi,:]
        _ = TB.solve_H()
        O_vals = np.zeros(len(points))

        
        for e in range(len(TB.Evec)):
                O_vals[e] = np.real(np.dot(np.conj(TB.Evec[e,:,inds[bi]]),np.dot(O,TB.Evec[e,:,inds[bi]])))
        Ovalues[:,bi] = O_vals
        p.append(ax.scatter(points[:,bi,0],points[:,bi,1],c=Ovalues[:,bi],cmap=cm.RdBu))
        if abs(Ovalues[:,bi]).max()>abs(Ovalues[:,cmappable]).max():
            cmappable = bi
                
    plt.colorbar(p[cmappable],ax=ax)
    plt.show()
        
    
        
        
    



if __name__ == "__main__":
    
    TB = PtCoO2._gen_TB()
    
    points = find_interceptions(TB,(16,17),1.5,200,20,0.0)
    
    O = operlib.S_vec(len(TB.basis),np.array([1,0,0]))
    operator(O,TB,(16,17),points)
    
    
            
                