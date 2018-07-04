# -*- coding: utf-8 -*-
"""
Created on Fri Feb 02 10:09:19 2018

@author: rday
MIT License

Copyright (c) 2018 Ryan Patrick Day

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

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
import ubc_tbarpes.TB_lib as TB_lib
import ubc_tbarpes.klib as klib

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

    

