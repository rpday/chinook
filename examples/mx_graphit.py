#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 20:00:10 2018

@author: ryanday
"""

import numpy as np
import matplotlib.pyplot as plt
import graphite
import ubc_tbarpes.build_lib as blib

def import_data(filename):
    data = []
    with open(filename,'r') as fromf:
        for l in fromf:
            tmp = l.split()
            if len(tmp)>0:
                if tmp[0] == '[':
                    data.append([float(t) for t in tmp[1:-2]])
    fromf.close()
    
    en = np.array(data[0])
    k = np.array(data[1])
    
    return en,k
                
        
        
        
if __name__=="__main__":
    
#    a,c =  2.46,3.35
#    avec = np.array([[np.sqrt(3)*a/2,a/2,0],
#                      [np.sqrt(3)*a/2,-a/2,0],
#                      [0,0,2*c]])
    
    txt='C:/Users/rday/Documents/TB_ARPES/2018/TB_ARPES_2018/TB_ARPES-master/examples/MDCfit_RD.txt'
    
    e,k=import_data(txt)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(k,e)

    kz = 2.7
    Kd = {'type':'A',
			'pts':[np.array([0,ki,kz]) for ki in k],#[np.array([0,1.702,2.1]),np.array([0,1.702,3.68]),np.array([0,1.702,4.75])],
			'grain':1,
            'labels':[]}
    TB = graphite.build_TB()
    
    TB.Kobj = blib.gen_K(Kd)
    TB.solve_H()
    for i in range(4):
        ax.plot(TB.Kobj.kpts[:,1],TB.Eband[:,i],c='r')
    
        