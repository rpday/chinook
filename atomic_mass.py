# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 09:32:57 2017

@author: rday
"""

import linecache
import pkg_resources



a_file = 'atomic_mass.txt'
filename = pkg_resources.resource_filename(__name__,a_file)
def get_mass_from_number(N_at):
    return float(linecache.getline(filename,int(N_at)).split('\t')[2][:-1])

def get_el_from_number(N_at):
    
    return linecache.getline(filename,int(N_at)).split('\t')[1]

def get_num_from_el(el):
    Z  = 0
    with open(filename,'r') as mass:
        for l in mass:
            line = l.split('\t')
            if line[1]==el:
                Z = int(line[0])
    mass.close()
    return Z