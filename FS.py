# -*- coding: utf-8 -*-
"""
Created on Fri Feb 02 10:09:19 2018

@author: rday
"""

'''

Fermi surface calculation

1. Begin by diagonalizing over a reasonably sized k-mesh, which extends beyond the first BZ
2. Find the points nearest and above and nearest below zero, linearly interpolate and find the point where crosses Ef
3. Organize crossings by band index -- these will form distinct pockets
4. If 'connected', find COM and then get the azimuthal angle around this COM--can then interpolate as a function of angle
    and establish a Fermi-surface contour
    If not 'connected', separate into distinct pockets and do the same as above
5. plot the contours from interpolation

'''


## to begin, I'll work on test case of a smaller region of k, where I have only Gamma-centred features. Then figure out how
## to treat pockets not centred at the origin.



import numpy as np
import matplotlib.pyplot as plt
import TB_lib
import klib

def coarse_FS(TB,x,y=None,z=0,E=0.0):
    xa = np.linspace(x[0],x[1],x[2])
    if y is not None:
        ya = np.linspace(y[0],y[1],y[2])
    else:
        ya = xa
    X,Y=np.meshgrid(xa,ya)    
    karr,_=klib.kmesh(0,X,Y,z)
    TB.Kobj = klib.kpath(karr)
    _,_=TB.solve_H()
    
    FS={}
    for p in range(len(karr)):
        for i in range(TB.basis):
            i
            try:
                FS[i].append([p/np.shape(X)[1],p%np.shape(X)[1]])
            except KeyError:
                FS[i]=[]
                FS[i].append([p/np.shape(X)[1],p%np.shape(X)[1]])
    return {f:np.array(FS[f]) for f in FS}

    

