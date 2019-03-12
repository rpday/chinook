# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 10:55:10 2019

@author: rday
"""

import numpy as np
import matplotlib.pyplot as plt
import chinook.rotation_lib as rotlib

hb = 6.626*10**-34/(2*np.pi)
c  = 3.0*10**8
q = 1.602*10**-19
A = 10.0**-10
me = 9.11*10**-31
mN = 1.67*10**-27
kb = 1.38*10**-23



def kvecs(Ek,th,ph):
    kn = np.sqrt(2*me/hb**2*(Ek)*q)*A
    return kn







if __name__ == "__main__":
    
    Ek = 16.7
    th = 0.0
    ph = np.linspace(-13,13,100)*np.pi/180.0
    
    kn = kvecs(Ek,th,ph)
