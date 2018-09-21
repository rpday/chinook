# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 09:39:22 2018

@author: rday
"""

'''
Different approaches to the matrix element calculation produce somewhat different results, unexpectedly. 
Two techniques however, e.r and A.Del give precisely the same result in angular coordinates, but vary in
terms of their radial component. The question remains then to explore whether these formally different radial
integrals produce meaningfully different results when computed. This script is intended to perform numerical integration
using my adaptive_int.py library to calculate the integrals under the two different approaches.

From e.r:
   /\ 
   |                3         
   |   dr j_l'(kr) r R_nl (r)
   |
  \/
  
And from A.Del:
    /\
    |              2
    |  dr j_l'(kr)r (dr - (2l-(2l+1)(l'-(l+1)))/2r)R_nl(r)
    |
   \/  

'''


import numpy as np
import matplotlib.pyplot as plt
import ubc_tbarpes.electron_configs as econ
import ubc_tbarpes.adaptive_int as adint
import ubc_tbarpes.atomic_mass as am
import scipy.special as sc

hb = 6.626*10**-34/(2*np.pi)
me = 9.11*10**-31
q = 1.602*10**-19
Eo = 8.85*10**-12
mp = 1.67*10**-27
c = 3.0*10**8
A = 10**-10

class orb:
    
    def __init__(self,Z,n,l):
        self.Z = Z
        self.n = n
        self.l = l
        self.mn = am.get_mass_from_number(self.Z)
        self.au = (4*np.pi*Eo*hb**2*(self.mn*mp+me))/(q**2*self.mn*mp*me)


'''
##################RADIAL WAVE FUNCTIONS############################################
'''        

def hydrogenic(r,Z,n,l,au):
    '''
    HYDROGENIC WAVEFUNCTION, SCALED TO TAKE R over ORDER 10 units (i.e. scaled to have r in units of Angstrom!)
    '''
    return (np.sqrt((2*Z/(n*au))**3*fact(n-l-1)/(2*n*fact(n+l)))*np.exp(-Z*r*A/(n*au))*(2*Z*r*A/(n*au))**l*sc.genlaguerre(n-l-1,l*2+1)(2*Z*r*A/(n*au)))

def d_hyd_r2(r,arglist):
    '''
    Derivative of HYDROGENIC WAVEFUNCTION. SINCE THIS IS FOR ONLY INTEGRAL PURPOSES, MULTIPLY BY R^2 to CIRCUMVENT OVERFLOW as R->0. IN ANY INTEGRAL,
    WE WILL HAVE THE R^2 FACTOR WHICH SAVES US from 1/R in the actual derivative on its own.
    NOTE R in units of ANGSTROM!
    The principal argument of the gen. Laguerre polynomial must be non-negative, so this term vanishes when n-l<2 and the derivative of the gen. Laguerre would otherwise have alpha<0
    
    '''
    n,l,Z,au = arglist[0],arglist[1],arglist[2],arglist[3]
    if (n-l<2):
        return (l*(A*r)-(A*r)**2*Z/(n*au))*hydrogenic(r,Z,n,l,au) 
    else:
        return (l*(A*r)-(A*r)**2*Z/(n*au))*hydrogenic(r,Z,n,l,au) - (A*r)**(2+l)*np.sqrt((2*Z/(n*au))**3*fact(n-l-1)/(2*n*fact(n+l)))*np.exp(-Z*(A*r)/(n*au))*(2*Z/(n*au))**(l+1)*sc.genlaguerre(n-l-2,l*2+2)(2*Z*A*r/(n*au))

'''
##################UTILITY FUNCTIONS############################################
'''

def fact(n):
    if n<0:
        return 0
    elif n<2:
        return 1
    else:
        return n*fact(n-1)


def hv_2_k(hv):
    return np.sqrt(2*me/hb**2*(hv)*q)*A

'''
##################INTEGRAND FUNCTIONS############################################
'''

def e_dot_r(r,arglist):
    '''
    Integrand for the e.r evaluation of the matrix element
    args:
        arglist -- list of [n,l,Z,au,lp,k]
        k -- float wavenumber
        r -- numpy float
        Z -- int atomic number
        n -- int prinicpal quantum number
        l -- int orbital quantum number
        au -- scaling factor for Hydrogenic orbital
        lp -- int final state orbital angular momentum (0,1,2,3)
    return -- numpy float

    For precision I have divided final result by A^3--just need to divide out same constant from both integrands to ensure numerical stability!
    '''
    n,l,Z,au,lp,k = arglist[0],arglist[1],arglist[2],arglist[3],arglist[4],arglist[5]
    return A*hydrogenic(r,Z,n,l,au)*r**3*(-1.0j)**lp*sc.spherical_jn(lp,k*r)



def e_dot_del(r,arglist):
    '''
    Integrand for the A.DEL evaluation of the matrix element
    args:
        arglist -- list of [n,l,Z,au,lp,k]
        k -- float wavenumber
        r -- numpy float
        Z -- int atomic number
        n -- int prinicpal quantum number
        l -- int orbital quantum number
        au -- scaling factor for Hydrogenic orbital
        lp -- int final state orbital angular momentum (0,1,2,3)
    return -- numpy float

    For precision I have divided final result by A--just need to divide out same constant from both integrands to ensure numerical stability!
    '''
    
    n,l,Z,au,lp,k = arglist[0],arglist[1],arglist[2],arglist[3],arglist[4],arglist[5]
    return (-1.0j)**lp*sc.spherical_jn(lp,k*r)*(d_hyd_r2(r,arglist)-(l if lp==l+1 else -(l+1))*hydrogenic(r,Z,n,l,au)*(A*r))
    
    
def e_dot_k(r,arglist):
    '''   
    Integrand for the A.DEL evaluation of the matrix element
    args:
        arglist -- list of [n,l,Z,au,lp,k]
        k -- float wavenumber
        r -- numpy float
        Z -- int atomic number
        n -- int prinicpal quantum number
        l -- int orbital quantum number
        au -- scaling factor for Hydrogenic orbital
        lp -- int final state orbital angular momentum (0,1,2,3)
    return -- numpy float

    For precision I have divided final result by A^3--just need to divide out same constant from both integrands to ensure numerical stability!
    '''
    n,l,Z,au,_,k = arglist[0],arglist[1],arglist[2],arglist[3],arglist[4],arglist[5]
    return A*(-1.0j)**l*(sc.spherical_jn(l,k*r))*hydrogenic(r,Z,n,l,au)*(r)**2
    
def _plt_vals(x,y):
    '''
    Simple plot routine for comparing a set of 1D arrays. Can pass:
        1.  1D x array and 1D y array
        2.  1D x array and 2D y array (with 2nd dimension matching length of x)
        3.  2D x array and 2D y array with x and y dim same
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if len(np.shape(y))==1:
        ax.plot(x,np.real(y))
        ax.plot(x,np.imag(y))
    elif len(np.shape(y))==2:
        if len(np.shape(x))==1:
            for i in range(np.shape(y)[0]):
                ax.plot(x,np.real(y[i,:]))
                ax.plot(x,np.imag(y[i,:]))
        elif np.shape(x)[0]==np.shape(y)[0]:
            for i in range(np.shape(y)[0]):
                ax.plot(x[i,:],np.real(y[i,:]))
                ax.plot(x[i,:],np.imag(y[i,:]))
    else:
        return False
    return True
    


if __name__=="__main__":
    
    Z,n,l=26,3,1
    
    Fe_3d = orb(Z,n,l)
    
    r0,rf,tol = 0,2,1e-2
    
    
    hv = np.linspace(1,200,80)
    ks = hv_2_k(hv)
    arglist = [Fe_3d.n,Fe_3d.l,Fe_3d.Z,Fe_3d.au,1,0]
    edr = np.zeros((len(hv),2),dtype=complex)
    edd = np.zeros((len(hv),2),dtype=complex)
    edk = np.zeros((len(hv),2),dtype=complex)
    for i in range(len(hv)):
        arglist[-1]=ks[i]
        
        edr[i,0] = me*hv[i]*1.0j*q/hb*(adint.integrate(e_dot_r,r0,rf,tol,[Fe_3d.n,Fe_3d.l,Fe_3d.Z,Fe_3d.au,l-1,ks[i]]))
        edd[i,0] = adint.integrate(e_dot_del,r0,rf,tol,[Fe_3d.n,Fe_3d.l,Fe_3d.Z,Fe_3d.au,l-1,ks[i]])
        edk[i,0] = abs(ks[i])*adint.integrate(e_dot_k,r0,rf,tol,[Fe_3d.n,Fe_3d.l,Fe_3d.Z,Fe_3d.au,l-1,ks[i]])
        
        edr[i,1] = A**4*me*hv[i]*1.0j*q/hb*(adint.integrate(e_dot_r,r0,rf,tol,[Fe_3d.n,Fe_3d.l,Fe_3d.Z,Fe_3d.au,l+1,ks[i]]))
        edd[i,1] = adint.integrate(e_dot_del,r0,rf,tol,[Fe_3d.n,Fe_3d.l,Fe_3d.Z,Fe_3d.au,l+1,ks[i]])
        edk[i,1] = edk[i,0]
        
        
    edr[:,0]/=abs(edr[:,0]).max()
    edr[:,1]/=abs(edr[:,1]).max()
    edd[:,0]/=abs(edd[:,0]).max()
    edd[:,1]/=abs(edd[:,1]).max()
    edk[:,0]/=abs(edk[:,0]).max()
    edk[:,1]/=abs(edk[:,1]).max()
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()
    plt.plot(hv[1:],abs((edr[1:,0])))
    plt.plot(hv[1:],abs((edd[1:,0])))
    plt.plot(hv[1:],abs(edk[1:,0]))
    plt.savefig('allforms.png')
    fig2 = plt.figure()
    
    ax = fig2.add_subplot(111)
    ax2 = ax.twinx()
    plt.plot(hv[1:],abs((edr[1:,1])))
    plt.plot(hv[1:],abs((edd[1:,1])))
    plt.plot(hv[1:],abs(edk[1:,1]))
    plt.savefig('allforms.png')
#        
#
