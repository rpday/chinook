# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 13:01:13 2018

@author: rday

SPECTRAL FUNCTIONS

"""
import numpy as np
import matplotlib.pyplot as plt
from ubc_tbarpes.ARPES_lib import con_ferm
hb = 6.626e-34/2/np.pi
q = 1.602e-19
me = 9.11e-31
A = 1e-10
kb = 1.38e-23

vf = np.vectorize(con_ferm)


def BCS_BEC(meff,k,w,G,D,ko,mu,T):
    p = hb**2/(2*me*q*A**2)
    xi = p*(k-ko)**2/meff+mu
    Ek = np.sqrt(xi**2+D**2)
    fk = vf(w/(kb*T/q))
    vk = np.sqrt(0.5*(1-xi/Ek))
    uk = np.sqrt(1-vk**2)
    K,W = np.meshgrid(k,w)
    Akw = np.zeros(np.shape(K))
    for i in range(np.shape(Akw)[1]):
        Akw[:,i]+=(uk[i]**2*G/(np.pi*((w-Ek[i])**2 + G**2)) + vk[i]**2*G/(np.pi*((w+Ek[i])**2+G**2)))*fk
    return K,W,Akw


def lin_Ak(meff,k,w,Go,Gl,ko,mu,T):
    p = hb**2/(2*me*q*A**2)
    Ek = p*(k-ko)**2/meff+mu
    fk = vf(w/(kb*T/q))
    G = Go + Gl*abs(w)
    Akw = np.zeros(np.shape(K))
    for i in range(np.shape(Akw)[1]):
        Akw[:,i]+=G/(np.pi*(w-Ek[i])**2+G**2)
        
    return K,W,Akw


def deplete(Akw,w,width,wo,scale):
    
    dA = np.zeros(np.shape(Akw))
    for i in range(np.shape(dA)[1]):
        dA[:,i] +=width/(np.pi*(w-wo)**2+width**2)
    dA/=abs(dA).max()
    return (np.ones(np.shape(dA)) - scale*dA)*Akw
    






if __name__ == "__main__":
    
    k = np.linspace(-0.25,1.25,500)
    w = np.linspace(-0.05,0.02,300)
    K,W = np.meshgrid(k,w)
    meff,G,D,koe,mue,T =3,0.003,0.01,1.0,-0.035,20
    Ak = np.zeros((300,500,10))

    Go,Gl,koh,muh = G,0.1,0.0,-0.015
    _,_,Ak2 = lin_Ak(-meff,k,w,Go,Gl,koh,muh,T)
    Ak2/=Ak2.max()
    
    for i in range(10):
    
        _,_,Ak[:,:,i] = BCS_BEC(meff,k,w,G,(i+1)*0.0015,koe,mue,T)
        Ak[:,:,i]/=Ak[:,:,i].max()
        
    
        
        Ak2p = deplete(Ak2,w,0.015,muh,(1-0.1*i))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.pcolormesh(K,W,Ak[:,:,i]+Ak2p,cmap=cm.magma)
        ax.set_xlabel('Momentum (1/A)')
        ax.set_ylabel('Energy (eV)')
        ax.set_title('Spectral Function: Gap {:0.01f} meV'.format((i+1)*1.5))
        plt.savefig('Akw_D_{:0.04f}.png'.format((i+1)*1.5))
        





