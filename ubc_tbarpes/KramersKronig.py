# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 09:47:44 2018

@author: rday
"""

from scipy.signal import hilbert
import numpy as np
import matplotlib.pyplot as plt





def im_linear(w,wc,yo,a,m):
    y = np.zeros(len(w),dtype=float)
    
    wc0 = np.where(abs(w+wc)==abs(w+wc).min())[0][0]
    wc1 = np.where(abs(w-wc)==abs(w-wc).min())[0][0]
    print(wc0,wc1,w[wc0],w[wc1])
    y[:wc0] = (m*abs(w[wc0])+yo)*np.exp(a*(w[:wc0]-w[wc0]))
    y[wc0:wc1] = m*abs(w[wc0:wc1])+yo
    y[wc1:] = (m*w[wc1]+yo)*np.exp(-a*(w[wc1:]-w[wc1]))
    return y


def KK(imE):
    out = hilbert(imE)
    return np.array([out.imag+1.0j*out.real])



def plot_KK(w,hilbert_T):
    fig=  plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(w,np.real(hilbert_T).T)
    ax.plot(w,np.imag(hilbert_T).T)
    
    

if __name__ == "__main__":
    
    m = -3
    wc = 3
    a = np.linspace(0.5,10,10)
    a = 50.0
    wc = np.linspace(120,180,10)
    w = np.linspace(-200,200,10000)
    yo = -0.5
#    hilbert_T = KK(im_linear(w,wc,yo,a,m))
#    plot_KK(w,hilbert_T)
    hilbert_T = np.zeros((len(wc),len(w)),dtype=complex)
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    for i in range(len(wc)):
        hilbert_T[i] = KK(im_linear(w,wc[i],yo,a,m))
        ax.plot(w,np.real(hilbert_T[i]))
        ax2.plot(w,np.imag(hilbert_T[i]))
        
    ax.set_xlabel('Energy (eV)')
    ax2.set_xlabel('Energy (eV)')
    ax.set_ylabel('Self Energy (eV)')
    ax.set_title('Re[Σ(k,w)]')
    ax2.set_title('Im[Σ(k,w)]')