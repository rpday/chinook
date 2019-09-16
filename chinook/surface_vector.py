# -*- coding: utf-8 -*-

#Created on Sun Jul 15 13:49:17 2018
#@author: ryanday
#MIT License

#Copyright (c) 2018 Ryan Patrick Day

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

import numpy as np

def ang_v1v2(v1,v2):
    '''
    Find angle between two vectors:
        
    *args*:

        - **v1**: numpy array of 3 float
        
        - **v2**: numpy array of 3 float
        
    *return*:

        - float, angle between the vectors
        
    ***
    '''
    
    return np.arccos(np.around(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)),15))

def are_parallel(v1,v2):
    
    '''
    Determine if two vectors are parallel:
        
    *args*:

        - **v1**: numpy array of 3 float
        
        - **v2**: numpy array of 3 float
        
    *return*:

        - boolean, True if parallel to within 1e-5 radians
   
    ***
    '''
    
    cos = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
    if abs(cos)>1e-5 and np.mod(cos,1)<1e-5:
        return True
    else:
        return False

def are_same(v1,v2):
    
    '''
    Determine if two vectors are identical
    
    *args*:

        - **v1**: numpy array of 3 float
        
        - **v2**: numpy array of 3 float
        
    *return*:

        - boolean, True if the two vectors are parallel and have same
        length, both to within 1e-5

    ***
    '''
    
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
    Seed search for v3 with the nearest-neighbouring Bravais latice point
    which maximizes the projection out of plane of that spanned by v1 and v2.
    
    *args*: 

        - **v1**, **v2**: numpy array of 3 float, the spanning vectors for plane 
    
        - **avec**: numpy array of 3x3 float
        
    *return*:

        - numpy array of 3 float, the nearby Bravais lattice point which
        maximizes the projection along the plane normal
        
    ***
    '''
    nv = np.cross(v1,v2)
    near = np.array([np.dot([int(np.mod(i,(3)))-1,int(np.mod(i/(3),(3)))-1,int(i/(3)**2)-1],avec) for i in range((3)**3) if i!=int((3)**3/2)])
    angs = np.array([ang_v1v2(ni,nv) for ni in near])
    choice = np.where(abs(angs)==abs(angs).min())[0]
    return near[choice[0]]


def refine_search(v3i,v1,v2,avec,maxlen):
    '''
    Refine the search for the optimal v3--supercell lattice vector which both 
    minimizes its length, while maximizing orthogonality with v1 and v2
    
    *args*:

        - **v3i**: numpy array of 3 float, initial guess for v3
        
        - **v1**: numpy array of 3 float, in-plane supercell vector
        
        - **v2**: numpy array of 3 float, in-plane supercell vector
        
        - **avec**: numpy array of 3x3 float, bulk lattice vectors
        
        - **maxlen**: float, upper limit on how long of a third vector we can
        reasonably tolerate. This becomes relevant for unusual Miller indices.
        
    *return*:

        - **v3_opt** list of numpy array of 3 float, list of viable options for 
        the out of plane surface unit cell vector
        
    ***
    '''
    nv = np.cross(v1,v2)
    atol = 1e-2
    v_add = np.array([-v2,-v1,np.zeros(3),v1,v2])
    v3_opt = []
    for qi in range(1,50):
        tmp_v = qi*v3i
        ang_to_norm = ang_v1v2(tmp_v,nv)
        if abs(ang_to_norm)<atol:
            ok = True
        else:
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
    
    '''
    To select the ideal out-of-plane surface unit cell vector, score the 
    candidates based on both their length and their orthogonality with respect
    to the two in-plane spanning vectors. The lowest scoring candidate is selected
    as the ideal choice.
    
    *args*:

        - **vlist**: list of len 3 numpy array of float, choices for out-of-plane
        vector
        
        - **v1**, **v2**: numpy array of 3 float, in plane spanning vectors
        
        - **avec**: numpy array of 3x3 float, primitive unit cell vectors
        
    *return*:

        - numpy array of len 3, out of plane surface-projected lattice vector
   
    ***
    '''
    nv = np.cross(v1,v2)
    
    angles = np.array([ang_v1v2(vi,nv) for vi in vlist])
    if abs(angles).max()>0.0:
        angles/=abs(angles).max()
    len_proj = np.array([np.linalg.norm(vi) for vi in vlist])
    if abs(len_proj).max()>0.0:
        len_proj/=abs(len_proj).max()
    score_vec = np.array([[len_proj[i],angles[i]] for i in range(len(vlist))])
    try:
        return vlist[np.where(np.linalg.norm(score_vec)==np.linalg.norm(score_vec).min())[0][0]]
    except IndexError:
        print('Require larger search field, increasing range of acceptance for surface lattice vectors.')
        return np.zeros(3)               

def find_v3(v1,v2,avec,maxlen):
    '''
    Find the best out-of-plane surface unit cell vector. While we initialize with
    a fixed cutoff for maximum length, to avoid endless searching, we can slowly 
    increase on each iteration until a good choice is possible.
    
    *args*:

        - **v1**, **v2**: numpy array of 3 float, in plane spanning vectors
        
        - **avec**: numpy array of 3x3 float, bulk lattice vectors
        
        - **maxlen**: float, max length tolerated for the vector we seek
        
    *return*:
    
        - **v3_choice**: the chosen unit cell vector
        
    ***
    '''
    
    v3i = initialize_search(v1,v2,avec)
    found = False
    while not found:
        v3f = refine_search(v3i,v1,v2,avec,maxlen)
        v3_choice = score(v3f,v1,v2,avec)
        if np.linalg.norm(v3_choice)>0:
            found = True
        else:
            maxlen *=1.2

    return v3_choice
                    
            
    
    
    


#if __name__ == "__main__":
#    avec = np.array([[0,4.736235,0],[4.101698,-2.368118,0],[0,0,13.492372]])
#    miller = np.array([1,0,4])
    
#    a =  3.52
#    avec = np.array([[a/2,a/2,0],[0,a/2,a/2],[a/2,0,a/2]])
#    miller = np.array([1,1,1])
#    v12 = v_vecs(miller,avec)
#    maxlen = 40
#    v3 = find_v3(v12[0],v12[1],avec,maxlen)