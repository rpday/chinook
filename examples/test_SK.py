# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 09:17:35 2018

@author: rday
"""

import numpy as np
import matplotlib.pyplot as plt
import ubc_tbarpes.plt_sph_harm as sph
import ubc_tbarpes.wigner as wig
import ubc_tbarpes.SK as sklib
import ubc_tbarpes.H_library as hlib
import ubc_tbarpes.rotation_lib as rlib
import ubc_tbarpes.Ylm as Ylm
import ubc_tbarpes.orbital as olib

normal_order = {0:{'':0},1:{'x':0,'y':1,'z':2},2:{'xz':0,'yz':1,'xy':2,'ZR':3,'XY':4},3:{'z3':0,'xz2':1,'yz2':2,'xzy':3,'zXY':4,'xXY':5,'yXY':6}}


def plot_path(o1,o2,Vllm):
    
    v1 = np.zeros(2*o1.l+1,dtype=complex)
    v1[normal_order[o1.l][o1.label[2:]]] = 1.0
    v2 = np.zeros(2*o2.l+1,dtype=complex)
    v2[normal_order[o2.l][o2.label[2:]]] = 1.0
    
    R12 = (o1.pos-o2.pos)
    Rmat = rlib.rotate_v1v2(R12,np.array([0,0,1]))
    A,B,y = rlib.Euler(Rmat)
#    print(A,B,y)
    RE = rlib.Euler_to_R(A,B,y)
    
    Ymats = Ylm.Yproj([o1,o2])
    
    o1rot = np.dot(wig.WignerD(o1.l,A,B,y),Ymats[(o1.atom,o1.n,o1.l,o1.spin)])
    o2rot = np.dot(wig.WignerD(o2.l,A,B,y),Ymats[(o2.atom,o2.n,o2.l,o2.spin)])
    
    ovec1 = np.dot((o1rot),v1)
    ovec2 = np.dot(o2rot,v2)
    
    CV = np.dot(RE,R12)
    
#    print('o1: ',np.around(ovec1,4))
#    print('o2: ',np.around(ovec2,4))
#    print('CV:',np.around(CV,4))
    
    
#    ro1 = np.dot(np.conj(Ymats[(o1.atom,o1.n,o1.l,o1.spin)]).T,np.conj(wig.WignerD(o1.l,A,B,y).T))
#    ro2 = np.dot(wig.WignerD(o2.l,A,B,y),Ymats[(o2.atom,o2.n,o2.l,o2.spin)])

    Vm = np.identity(3)*Vllm
#    
    
    
    print('SK\n',np.around(np.dot(np.conj(o1rot.T),np.dot(Vm,o2rot)),4))
    
    
    
    _ = sph.plot_psi(15,[o1,o2],np.array([0.707,0.707]))
    
    o1.proj = vector_to_proj(ovec1)
    o2.proj = vector_to_proj(ovec2)
    o1.pos = CV
    
    _ = sph.plot_psi(15,[o1,o2],np.array([0.707,0.707]))
    return ovec1,ovec2
    


def vector_to_proj(vec):
    proj = []
    l = int((len(vec)-1)/2)
    for vi in range(len(vec)):
        if abs(vec[vi])>1e-10:
            proj.append([np.real(vec[vi]),np.imag(vec[vi]),l,l-vi])
    return np.array(proj)
    


def SKmat(basis):
    SKfuncs = sklib.SK_full(basis)
    
    return SKfuncs


def SK_test(SKfunc,vec,V):
    A,B,y = rlib.Euler(rlib.rotate_v1v2(vec,np.array([0,0,1])))
    
    
    print(np.around(SKfunc(A,B,y,V),4))
    
    
    
def pp(R,s,p):
    R/=np.linalg.norm(R)
    l,m,n = R[0],R[1],R[2]
    V = np.array([[l**2*s+(1-l**2)*p,l*m*(s-p),l*n*(s-p)],[l*m*(s-p),m**2*s+(1-m**2)*p,m*n*(s-p)],[l*n*(s-p),m*n*(s-p),n**2*s+(1-n**2)*p]])
    return V


def sp(R,s):
    R/=np.linalg.norm(R)
    l,m,n = R[0],R[1],R[2]
    V = np.array([[l*s,m*s,n*s]])
    return V


if __name__ == "__main__":
    
    pos0 = np.zeros(3)
    pos1 = np.array([1,0.5,0.2])
    
    
    o1 = olib.orbital(0,0,'21x',pos1,6)
    o2 = olib.orbital(0,1,'21x',pos0,6)
    o3 = olib.orbital(0,2,'20',pos0,6)
    
    SKfuncs = SKmat([o1,o2,o3])
    Vs,Vp = 2.,1.
    Vllm = [Vp,Vs,Vp]
    print('test SK')
    SK_test(SKfuncs[(0,0,2,2,1,1)],pos1,Vllm)
    print('\n')
    
    
    plot_path(o1,o2,Vllm)
    
    ppm=pp(pos1,Vs,Vp)
    print('pp\n')
    print(np.around(ppm,4))
    
    SKfuncs
    
    SK_test(SKfuncs[(0,0,2,2,0,1)],pos1,[Vs])
    
    spm = sp(pos1,Vs)
    print('sp\n')
    print(np.around(spm,4))
    