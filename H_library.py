#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 21:38:24 2017

@author: ryanday

Slater Koster Library for generating a Hamiltonian of Slater-Koster
"""

import numpy as np
import orbital as orb
import SK

hb = 6.626*10**-34/(2*np.pi)
c  = 3.0*10**8
q = 1.602*10**-19
A = 10.0**-10
me = 9.11*10**-31
mn = 1.67*10**-27
kb = 1.38*10**-23


#def slabify():
    

def txt_build(filename,cutoff,renorm,offset,tol):
    
    Hlist = []
    
    with open(filename,'r') as origin:
        
        for line in origin:
            
            spl = line.split(',')
            R = np.array([float(spl[2]),float(spl[3]),float(spl[4])])
            Hval = complex(spl[5])
            
            if len(spl)>7:
                Hval+=1.0j*float(spl[6])
            if abs(Hval)>tol and  np.linalg.norm(R)<cutoff:
                Hval*=renorm
                if np.linalg.norm(R)==0.0:
                    Hval-=offset
                    
                tmp = [int(spl[0]),int(spl[1]),R[0],R[1],R[2],Hval]

                Hlist.append(tmp)
            
    origin.close()
            
    return Hlist


                

def sk_build(cluster,V,cutoff,tol):
    '''
    Would like to find a better way of doing this, or at least getting around the whole cluster thing...
    '''
    
    cutoff = [0.0]+cutoff
    H_raw = []
    o1o2norm = {}
    
    for o1 in cluster:
        for o2 in cluster:
            dir_cos = np.zeros(3)
            
            Rij = o2.pos-o1.pos
            
            Rijn = np.linalg.norm(Rij)
            
            if Rijn>0.0:
                dir_cos = np.copy(Rij)#/np.linalg.norm(o2.pos-o1.pos)

            dirstr = "-{:0.2f}-{:0.2f}-{:0.2f}".format(dir_cos[0],dir_cos[1],dir_cos[2])
            orb_label = str(o1.tag)+'-'+str(o2.tag)+'-'+dirstr
            if Rijn>max(cutoff):

                o1o2norm[orb_label] = True
                
            elif o2.index>=o1.index: #only going through the j>=i elements--saves computational time o2.tag>=o1.tag and
                try:
                    o1o2norm[orb_label] #check to see if a pairing of these orbitals, along this direction, has already been calculated--if so, skip to next o1,o2                     
                    continue
                except KeyError: #if this pair has not yet been included, proceed

                    if isinstance(V,list): #if we have given a list of SK dictionaries (relevant to different distance ranges)                        
                        for i in range(len(cutoff)-1):
                            if cutoff[i]<Rijn<cutoff[i+1]: #if in this range of the dictionaries, use the lower bound
                                tmp_V = V[i]
                                mat_el = SK.SK_coeff(o1,o2,tmp_V)  #then matrix element is computed using the SK function
                    elif isinstance(V,dict): #if the SK matrix elements brought in NOT as a list of dictionaries...                               
                        mat_el = SK.SK_coeff(o1,o2,V)

                            
                    if abs(mat_el)>tol:                           
                        H_raw.append([o1.tag,o2.tag,Rij[0],Rij[1],Rij[2],mat_el])
                            
                    o1o2norm[orb_label] = True #now that the pair has been calculated, disqualify from subsequent calculations
    return H_raw


def spin_double(H,lb):
    lenb = int(lb/2)
    h2 = []
    for i in range(len(H)):
        h2.append([H[i][0]+lenb,H[i][1]+lenb,H[i][2],H[i][3],H[i][4],H[i][5]])
    return h2



def SO(basis,Md):
    normal_order = {1:{'x':0,'y':1,'z':2},2:{'xz':0,'yz':1,'xy':2,'ZR':3,'XY':4}}
    LS = {}
    al = []
    HSO = []
    for o in basis:
        if (o.atom,o.l) not in al:
            M = Md[(o.atom,o.l)]
            LS[(o.atom,o.l)] = LSmat(M,o.l)
    for o1 in basis:
        for o2 in basis:
            if np.linalg.norm(o1.pos-o2.pos)<0.0001 and o1.l==o2.l:
                L = o1.lam
                inds = (normal_order[o1.l][o1.label[2:]],normal_order[o2.l][o2.label[2:]]) 
        
                ds = (o1.spin-o2.spin)/2.0
                if ds ==1.0:
                    LSm = LS[(o1.atom,o1.l)]['+']
                elif ds ==-1.0:
                    LSm = LS[(o1.atom,o1.l)]['-']
                elif ds==0.0:
                    if o1.spin==1:
                        LSm = LS[(o1.atom,o1.l)]['u']
                    else:
                        LSm = LS[(o1.atom,o1.l)]['d']
                if abs(LSm[inds])!=0:
                    HSO.append([o1.index,o2.index,0,0,0,L*LSm[inds]])
        
    return HSO
    
def LSmat(M,l):
    Mp = np.linalg.inv(M)
    if l==1:
        LpSm = np.dot(Mp,np.dot(LS_1([1,0,0]),M))
        LmSp = np.dot(Mp,np.dot(LS_1([0,0,1]),M))
        LzSzd = np.dot(Mp,np.dot(LS_1([0,-0.5,0]),M))
        LzSzu = np.dot(Mp,np.dot(LS_1([0,0.5,0]),M))
    elif l==2:
        LpSm = np.dot(Mp,np.dot(LS_2([1,0,0]),M))
        LmSp = np.dot(Mp,np.dot(LS_2([0,0,1]),M))
        LzSzd = np.dot(Mp,np.dot(LS_2([0,-0.5,0]),M))
        LzSzu = np.dot(Mp,np.dot(LS_2([0,0.5,0]),M))
    return {'-':LpSm,'+':LmSp,'d':LzSzd,'u':LzSzu}
            

def LS_1(s):
    return np.array([[s[1],np.sqrt(0.5)*s[0],0.0],[np.sqrt(0.5)*s[2],0.0,np.sqrt(0.5)*s[0]],[0.0,np.sqrt(0.5)*s[2],-s[1]]])            
def LS_2(s):
    return np.array([[2*s[1],s[0],0,0,0],[s[2],s[1],np.sqrt(1.5)*s[0],0,0],[0,np.sqrt(1.5)*s[2],0,np.sqrt(1.5)*s[0],0],[0,0,np.sqrt(1.5)*s[2],-s[1],s[0]],[0,0,0,s[2],-2*s[1]]])  





def Yproj(basis):
    '''
    Define the unitary transformation rotating the basis of different inequivalent atoms in the
    basis to the basis of spherical harmonics for sake of defining L.S operator in basis of user
    args: basis--list of orbital objects
    
    returns: dictionary of matrices for the different atoms and l-shells--keys are tuples of (atom,l)
    
    Note this works only on p and d type orbitals, s is irrelevant, not currently supporting f orbitals
    
    '''
    normal_order = {1:{'x':0,'y':1,'z':2},2:{'xz':0,'yz':1,'xy':2,'ZR':3,'XY':4},3:{'z3':0,'xz2':1,'yz2':2,'xzy':3,'zXY':4,'xXY':5,'yXY':6}}
    a = basis[0].atom
    l = basis[0].l
    M = {}
    M_tmp = np.zeros((2*l+1,2*l+1),dtype=complex)
    for b in basis:
        if b.atom==a and b.l==l:
            label = b.label[2:]
            for p in b.proj:
                M_tmp[l-int(p[-1]),normal_order[l][label]] = p[0]+1.0j*p[1]
            
        else:
            #If we are using a reduced basis, fill in orthonormalized projections for other states in the shell
            #which have been ignored in our basis choice--these will still be relevant to the definition of the LS operator
            M_tmp = fillin(M_tmp,l)            
            M[(a,l)] = M_tmp
            ##Initialize the next M matrix               
            a = b.atom
            l = b.l
            M_tmp = np.zeros((2*l+1,2*l+1),dtype=complex)
            
    M_tmp = fillin(M_tmp,l)
    M[(a,l)] = M_tmp
    
    return M

def fillin(M,l):
    normal_order_rev = {1:{0:'x',1:'y',2:'z'},2:{0:'xz',1:'yz',2:'xy',3:'ZR',4:'XY'},3:{'z3':0,'xz2':1,'yz2':2,'xzy':3,'zXY':4,'xXY':5,'yXY':6}}

    for m in range(2*l+1):
        if np.linalg.norm(M[:,m])==0: #if column is empty (i.e. user-defined projection does not exist)
            proj = np.zeros(2*l+1,dtype=complex) 
            for pi in orb.projdict[str(l)+normal_order_rev[l][m]]: 
                proj[l-int(pi[-1])] = pi[0]+1.0j*pi[1] #fill the column with generic projection for this orbital (this will be a dummy)
            for mp in range(2*l+1): #Orthogonalize against the user-defined projections
                if np.linalg.norm(M[:,mp])!=0:
                    proj = GrahamSchmidt(proj,M[:,mp])
            M[:,m] = proj            
    return M


def GrahamSchmidt(a,b):
    '''
    Simple orthogonalization of two vectors, returns orthonormalized vector
    args: a,b -- np.array of same length
    returns: tmp -- numpy array of same size, orthonormalized to the b vector
    '''
    tmp = a - np.dot(a,b)/np.dot(b,b)*b
    return tmp/np.linalg.norm(tmp)
        
        
        
    

    

