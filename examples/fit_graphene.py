# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 13:55:48 2018

@author: rday
"""

import sys
sys.path.append('C:/Users/rday/Documents/TB_ARPES/2018/TB_ARPES_2018/TB_ARPES-master/')

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import graphene_qespresso as graphene
import load_bands as qe



def load_TB():
    
    return graphene.gen_model()




if __name__ == "__main__":
    
    TB = load_TB()
    