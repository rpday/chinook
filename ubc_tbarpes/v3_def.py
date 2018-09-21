# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 13:49:17 2018

@author: rday

Define out-of-plane slab vector
"""

import numpy as np
import ubc_tbarpes.build_lib
import ubc_tbarpes.slab

def ang_v1v2(v1,v2):
    return np.arccos(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))

def are_parallel(v1,v2):
    
    cos = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
    if abs(cos)>1e-5 and np.mod(cos,1)<1e-5:
        return True
    else:
        return False

def are_same(v1,v2):
    
    if are_parallel(v1,v2):
        mod_val = (np.mod(np.dot(v1,v2)/np.linalg.norm(v2)**2,1))
        if mod_val<1.0e-5 or mod_val>0.99999:
            return True
        else:
            return False
    else:
        return False
    
    
def initialize_search(v1,v2,avec):
    '''
    Seed search for v3 with the nearest-neighbouring Bravais latice point which maximizes the projection
    out of plane of that spanned by v1 and v2
    args: v1,v2 -- the spanning vectors for plane numpy array of 3 float
    avec -- numpy array of 3x3 float
    return: the nearby Bravais lattice point which maximizes the projection along the plane normal
    '''
    nv = np.cross(v1,v2)
    near = np.array([np.dot([int(np.mod(i,(3)))-1,int(np.mod(i/(3),(3)))-1,int(i/(3)**2)-1],avec) for i in range((3)**3) if i!=int((3)**3/2)])
    angs = np.array([ang_v1v2(ni,nv) for ni in near])
    choice = np.where(abs(angs)==abs(angs).min())[0]
    return near[choice[0]]


def refine_search(v3i,v1,v2,avec,maxlen):
    '''
    Refine the search for the optimal v3 which both minimalizes the length while maximizing orthogonality
    to v1 and v2
    '''
    nv = np.cross(v1,v2)
    atol = 1e-2
    v_add = np.array([-v2,-v1,np.zeros(3),v1,v2])
    v3_opt = []
    for qi in range(1,50):
        tmp_v = qi*v3i
        ang_to_norm = ang_v1v2(tmp_v,nv)
        ok = False
        counter = 1
        while not ok:
            v_opt = tmp_v + counter*v_add
            angles = np.array([ang_v1v2(vi,nv) for vi in v_opt])
            try:
                better = abs(angles[np.where(abs(angles)<abs(ang_to_norm))[0]]).min()
                tmp_v = v_opt[np.where(abs(angles)==better)[0][0]]
                ang_to_norm = ang_v1v2(tmp_v,nv)
                if abs(ang_v1v2(tmp_v,nv))<atol:
                    ok = True

            except ValueError:
                ok = True
            counter+=1
        if np.linalg.norm(tmp_v)<maxlen:
            v3_opt.append(tmp_v)
    return v3_opt

def score(vlist,v1,v2,avec):
    nv = np.cross(v1,v2)
    
    angles = np.array([ang_v1v2(vi,nv) for vi in vlist])
    if abs(angles).max()>0.0:
        angles/=abs(angles).max()
    len_proj = np.array([np.linalg.norm(vi) for vi in vlist])
    if abs(len_proj).max()>0.0:
        len_proj/=abs(len_proj).max()
    score_vec = np.array([[len_proj[i],angles[i]] for i in range(len(vlist))])
    return vlist[np.where(np.linalg.norm(score_vec)==np.linalg.norm(score_vec).min())[0][0]]
                    

def find_v3(v1,v2,avec,maxlen):
    v3i = initialize_search(v1,v2,avec)
    v3f = refine_search(v3i,v1,v2,avec,maxlen)
    v3_choice = score(v3f,v1,v2,avec)
    return v3_choice
                    
            
    
    
    


if __name__ == "__main__":
#    avec = np.array([[0,4.736235,0],[4.101698,-2.368118,0],[0,0,13.492372]])
#    miller = np.array([1,0,4])
    
    a =  3.52
    avec = np.array([[a/2,a/2,0],[0,a/2,a/2],[a/2,0,a/2]])
    miller = np.array([1,1,1])
    v12 = slab.v_vecs(miller,avec)
    maxlen = 40
    v3 = find_v3(v12[0],v12[1],avec,maxlen)