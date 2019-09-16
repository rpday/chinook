# -*- coding: utf-8 -*-

#Created on Sun Jul 15 13:49:17 2018


#@author: rday

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
#The following algorithm is based very closely off of the work of 
#W. Sun and G. Ceder, Surface Science 617, 53 (2013)

import numpy as np

def ang_v1v2(v1,v2):
    '''
    Find angle between two vectors, rounded to floating point precision.

    *args*:

        - **v1**: numpy array of N float

        - **v2**: numpy array of N float

    *return*: 

        - float, angle in radians

    ***
    '''
    return np.arccos(np.around(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)),15))

def are_parallel(v1,v2):
    '''
    Are two vectors parallel?

    *args*:

        - **v1**: numpy array of N float

        - **v2**: numpy array of N float

    *return*:

        - boolean, True if parallel, to within 1e-5 radians, False otherwise

    ***
    '''
    cos = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
    if abs(cos)>1e-5 and np.mod(cos,1)<1e-5:
        return True
    else:
        return False

def are_same(v1,v2):
  
    '''
    Are two vectors identical, i.e. parallel and of same length, to within 
    the precision of *are_parallel*?

    *args*:

        - **v1**: numpy array of N float

        - **v2**: numpy array of N float

    *return*:

        - boolean, True if identical, False otherwise.

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
    Seed search for v3 with the nearest-neighbouring Bravais latice point which maximizes
    the projection out of plane of that spanned by v1 and v2

    *args*:

        - **v1**: numpy array of 3 float, a spanning vector of the plane

        - **v2**: numpy array of 3 float, a spanning vector of the plane
    
        - **avec**: numpy array of 3x3 float

    *return*:

        - numpy array of float, the nearby Bravais lattice point which maximizes
         the projection along the plane normal

    ***
    '''
    nv = np.cross(v1,v2)
    near = np.array([np.dot([int(np.mod(i,(3)))-1,int(np.mod(i/(3),(3)))-1,int(i/(3)**2)-1],avec) for i in range((3)**3) if i!=int((3)**3/2)])
    angs = np.array([ang_v1v2(ni,nv) for ni in near])
    choice = np.where(abs(angs)==abs(angs).min())[0]
    return near[choice[0]]


def refine_search(v3i,v1,v2,avec,maxlen):
    '''
    Refine the search for the optimal v3 which both minimalizes the length while
    maximizing orthogonality to v1 and v2

    *args*:

        - **v3i**: numpy array of 3 float, initial guess for the surface vector

        - **v1**: numpy array of 3 float, a spanning vector of the plane

        - **v2**: numpy array of 3 float, a spanning vector of the plane

        - **avec**: numpy array of 3x3 float

        - **maxlen**: float, longest vector accepted

    *return*:

        **v3_opt**: list of numpy array of 3 float, options for surface vector
    
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

    The possible surface vectors are scored based on their legnth and their orthogonality 
    to the in-plane vectors.

    *args*:

        - **vlist**: list fo numpy array of 3 float, options for surface vector

        - **v1**: numpy array of 3 float, a spanning vector of the plane

        - **v2**: numpy array of 3 float, a spanning vector of the plane

        - **avec**: numpy array of 3x3 float

    *return*:

        - numpy array of 3 float, the best scoring vector option

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
    return vlist[np.where(np.linalg.norm(score_vec)==np.linalg.norm(score_vec).min())[0][0]]
                    

def find_v3(v1,v2,avec,maxlen):
    '''
    Wrapper function for finding the surface vector. 

    *args*:

        - **v1**: numpy array of 3 float, a spanning vector of the plane

        - **v2**: numpy array of 3 float, a spanning vector of the plane

        - **avec**: numpy array of 3x3 float

        - **maxlen**: float, longest accepted surface vector
        
    *return*: 

        - numpy array of 3 float, surface vector choice  

    ***  
    '''
    v3i = initialize_search(v1,v2,avec)
    v3f = refine_search(v3i,v1,v2,avec,maxlen)
    v3_choice = score(v3f,v1,v2,avec)
    return v3_choice
                    
            
    