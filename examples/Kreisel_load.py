# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 08:39:25 2018

@author: rday
"""

import numpy as np


def load_file(fnm):
    data = []
    with open(fnm,'r') as fromfile:
        for line in fromfile:
            data.append([float(ti) for ti in line.split(',')])
            
    return np.array(data)



def data_to_Fe(data,avec,pos):
    Hlist = []
    for d in data:
        vec = np.dot(np.array(d[:3]),avec)
        xtra = pos[int((d[4]-1)/5)]-pos[int((d[3]-1)/5)]
        tmp = [int(d[3]-1),int(d[4]-1),*vec+xtra,d[5]+1.0j*d[6]]
        Hlist.append(tmp)
        
    return Hlist

def transform(Helements):
    tmp = []
    inds = np.array([1.0,1.0,-1.0j,-1.0j,1.0,1.0,1.0,1.0j,1.0j,1.0],dtype=complex)
    for h in Helements:
        hval = np.conj(inds[int(h[0])])*complex(h[-1])*inds[int(h[1])]    
        new_val = [*h[:5],hval]
        tmp.append(new_val)
    return tmp


def mod_model(H):
    '''
    Kreisel's model includes OO within the kinetic Hamiltonian--remove these terms.
    Also adjust the dxy, and eg band positions to match ARPES better
    '''
    tmp_H = []
    for hi in H:
        hii = hi
        if np.mod(hi[0],5)==0 and np.sqrt(hi[2]**2+hi[3]**2+hi[4]**2)==0.0:
            hii[-1]+=0.04
        if (np.mod(hi[0],5)==1 or np.mod(hi[0],5)==4) and np.sqrt(hi[2]**2+hi[3]**2+hi[4]**2)==0.0:
            hii[-1]-=0.06
        
        tmp_H.append(hii)
        
    return tmp_H



def gen_Kreisel_list(avec,pos):
    
    fnm = 'C:/Users/rday/Documents/TB_ARPES/2018/TB_ARPES_2018/FeSe_SO_OO/Kriesel/tb_FeSe_generic_no_soc10_band_.csv'
    data = load_file(fnm)
    
    Fe1,Fe2 = pos[0],pos[1]
    
    Hl = data_to_Fe(data,avec,[Fe1,Fe2])
    
    return transform(Hl)
            
def write_H(filename,H):
    
    with open(filename,'w') as tofile:
        for hi in H:
            tofile.write('{:d},{:d},{:0.05f},{:0.05f},{:0.05f},{:0.08f}\n'.format(*hi[:-1],np.real(hi[-1])))
    tofile.close()



if __name__=="__main__":
    a,b,c = 2.665,2.655,5.48
    a,b,c = 2.66,2.66,5.48
    avec = np.array([[a,b,0.0],[a,-b,0.0],[0.0,0.0,c]])
    Fe1,Fe2 = np.array([0,0,0]),np.array([a,0,0])
    
#    avec = np.array([[a/np.sqrt(2),a/np.sqrt(2),0.0],[-a/np.sqrt(2),a/np.sqrt(2),0.0],[0.0,0.0,c]])
#    Fe1,Fe2 = np.array([-a/np.sqrt(8),0,0]),np.array([a/np.sqrt(8),0,0])
    
    H = gen_Kreisel_list(avec,[Fe1,Fe2])
    H = mod_model(H)
    write_H('FeSe_Kreisel_mod.txt',H)