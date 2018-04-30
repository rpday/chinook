# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 10:15:56 2016

@author: rday

MIT License

Copyright (c) 2018 Ryan Patrick Day

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""


'''
Slater's rules for calculating effective nuclear charge:
    Group --- Other electrons in --- electrons in group(s) w/ --- electrons in group(s) --- electrons in group(s) w/
                the same group            n'=n, l'<l                   w/ n'=n-1                    n'<=n-2
 -------------------------------------------------------------------------------------------------------------------
  [1s]    ---        0.30        ---         -                ---          -            ---            -
  [ns,np] ---        0.35        ---         -                ---         0.85          ---            1
  [nd,nf] ---        0.35        ---         1                ---          1            ---            1

Effective nuclear charge: n = 1,2,3,4,5,6 --- n' = 1,2,3,3.7,4.0,4.2


'''

import numpy as np
import scipy.special as sc
import matplotlib.pyplot as plt
import atomic_mass as am
from math import factorial
from math import gamma
import Ylm as Y

ao = 5.29177*10**-1 #in units of angstroms
me = 9.11*10**-31
mp = 1.67*10**-27


filename="electron_configs.txt"
prefs={"[He]":1,"[Ne]":9,"[Ar]":17,"[Kr]":35,"[Xe]":53,"[Rn]":85}

Z=[]
cons = []
e_con=open(filename,"r")
for line in e_con:
    read = line.split()
    Z.append(int(read[0]))
    try:
        tmp = prefs[read[2][:4]]
        cons.append(cons[tmp]+read[2][4:])
    except KeyError:
        cons.append(read[2])
        
def shield_split(shield_string):
    alphas = []
    for s in range(len(shield_string)):
        if shield_string[s].isalpha():
            alphas.append(s)
    parsed_shield = []
    for s in range(len(alphas)-1):
        parsed_shield.append(shield_string[alphas[s]-1:alphas[s+1]-1])
    parsed_shield.append(shield_string[alphas[-1]-1:])
    return parsed_shield
        
def Z_eff(Z_ind,orb):
    l_dic = {"s":0,"p":1,"d":2,"f":3}  
    l_dic_inv = {0:"s",1:"p",2:"d",3:"f"}
    e_conf = cons[Z_ind-1]
    n = int(orb[0])
    if orb[1].isalpha():
        l=orb[1] # if orbital string already written in form of 2p for example, rather than 21, simply take l as 'p'
    elif not orb[1].isalpha():#if its not written as 2p, but rather 21, then convert the 1 into p and then rewrite orb as '2p'
        l=int(orb[1])
        tmp = orb[0]+l_dic_inv[l]
        orb = tmp
    for s in range(len(e_conf)-1): #iterate over the character string for the electronic configuration from table
        if e_conf[s]+e_conf[s+1]==orb: #if we find the orbital string, stop
            shield=e_conf[:s] #take the shield as all preceding elements of the configuration
            val = e_conf[s:s+3] # the valence are the remaining
            if orb[-1]=="s": #for the case of an s valence, then you need to add the next set of stuff too
                shield+=e_conf[s+3:s+6]
#    print 'shield shells',shield
#    print 'valence',val
    parsed_shield=shield_split(shield)
    nval = int(val[0])
    lval = l_dic[val[1]]
    fill_val = int(val[2:])
    s_sum = 0.0
    if nval==1:
        s_sum+=0.3*(fill_val-1)
    if nval>1:
        for i in range(len(parsed_shield)):
            tmp = parsed_shield[i]
            n = int(tmp[0])
            l = l_dic[tmp[1]]
            fill = int(tmp[2:])
            if n<=nval-2:
                s_sum+=1.0*fill
            elif n==nval-1:
                if lval<=1:
                    s_sum+=0.85*fill
                elif lval>1:
                    s_sum+=1.0*fill
            if n==nval and l<lval and lval>1:
                s_sum+=1.0*fill
            elif n==nval and lval<2:
                s_sum+=0.35*fill
        s_sum+=0.35*(fill_val-1) #using Slater's rules from above to compute the effective screening
    result = Z_ind-s_sum
    return result
    
def n_eff(n):
    ndic={1:1,2:2,3:3,4:3.7,5:4,6:4.2}
    return ndic[n]

def Slater(Z_ind,orb,r):
    ne = n_eff(int(orb[0]))
    xi = Z_eff(Z_ind,orb)/ne#/0.5
    tmp = (2*xi)**ne*np.sqrt(2*xi/gamma(float(2*ne)+1))
    result = tmp*r**(ne-1)*np.exp(-xi*r)
    return result,xi,ne
  
def hydrogenic(Z_ind,mn,orb,r):
    mu = me*mn*mp/(me+mn*mp)
    au = me/mu*ao
    n = int(orb[0])
    l = int(orb[1])
    root = np.sqrt((2*Z_ind/(n*au))**3*factorial(n-l-1)/float(factorial(n+l))/(2.*n))
#    print mu,au,n,l,root
    orb = root*np.exp(-Z_ind*2*r/(n*au))*(2*Z_ind*r/(n*au))**l*Y.laguerre(r*2*Z_ind/(n*au),2*l+1,n-l-1)
    return orb

if __name__=="__main__":
    mn = am.get_mass_from_number(6)
    r = np.linspace(0,10,1000)
    Chyd = hydrogenic(6,mn,'21z',r)
    CSlat,_,_ = Slater(6,'21x',r)
    plt.figure()
    plt.plot(r,Chyd)
    plt.plot(r,CSlat)
    
#    vals = Slater(6,'21x',0.1)

    
    #        print "n",n,"l",l,"fill",fill
    