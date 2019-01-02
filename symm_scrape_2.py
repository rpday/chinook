#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 14:17:57 2018

@author: ryanday
"""

import urllib3



def scrape_all():
    '''
    Scrape the symmetry operation definitions from the University College, London
    and 
    '''
    http = urllib3.PoolManager()
    ops = []
    names = []
    for i in range(1,231):
        r = http.request('GET','http://img.chem.ucl.ac.uk/sgp/LARGE/{:03d}az3.htm'.format(i))
        if r.status==200:
            byte_str = str(r.data)
            name_start = byte_str.find('Space Group Symbol')+18
            name_stop = byte_str[name_start:name_start+30].find('"')
            name = byte_str[name_start:name_start+name_stop]
            
            start,stop = byte_str.find('<PRE>'),byte_str.find('</PRE>')
            sub_byte = byte_str[start+5:stop].replace('\\r','').split('\\n')[1:-1]
            clean_byte = [s.upper().strip().replace(' ','') for s in sub_byte]
            ultra_clean_byte = [s for s in clean_byte if len(s)>0]
            ops.append(ultra_clean_byte)
            names.append(name)
    return ops,names


def save_symm_ops(ops,names,filename):
    
    with open(filename,'w') as tofile:
        for o in list(enumerate(ops)):
            write_ln = 'Group #{:d}: {:s}\n'.format(o[0]+1,names[o[0]])
            for oi in o[1]:
                write_ln+=oi+'\n'
            write_ln+='\n'
            tofile.write(write_ln)
    tofile.close()
    print('Printed')
            
            
if __name__ == "__main__":

    ops,names = scrape_all()    
    save_symm_ops(ops,names,'symmetry_group_operations.txt')     
