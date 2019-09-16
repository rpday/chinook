# -*- coding: utf-8 -*-

#Created on Wed Sep 28 10:15:56 2016

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



#Slater's rules for calculating effective nuclear charge:
#   Group --- Other electrons in --- electrons in group(s) w/ --- electrons in group(s) --- electrons in group(s) w/
#               the same group            n'=n, l'<l                   w/ n'=n-1                    n'<=n-2
# -------------------------------------------------------------------------------------------------------------------
#  [1s]    ---        0.30        ---         -                ---          -            ---            -
#  [ns,np] ---        0.35        ---         -                ---         0.85          ---            1
#  [nd,nf] ---        0.35        ---         1                ---          1            ---            1

#Effective nuclear charge: n = 1,2,3,4,5,6 --- n' = 1,2,3,3.7,4.0,4.2


import numpy as np
import scipy.special as sc
import matplotlib.pyplot as plt
import pkg_resources
import linecache
from math import factorial
from math import gamma

import chinook.Ylm as Y
import chinook.atomic_mass as am

ao = 5.29177*10**-1 #in units of angstroms
me = 9.11*10**-31
hb = 6.626e-34/2/np.pi
q = 1.602e-19
Eo = 8.85e-12
mp = 1.67*10**-27
A = 10**-10

ndic={1:1,2:2,3:3,4:3.7,5:4,6:4.2}


textnm="electron_configs.txt"

filename = pkg_resources.resource_filename(__name__,textnm)

def get_con(filename,Z):
    
    '''
    Get electron configuration for a given element, from electron_configs.txt.
    This is configuration to be used in calculation of the Slater wavefunction.
    
    *args*:

        - **filename**: string, text-file where the configurations are saved
        
        - **Z**: int, atomic number
        
    *return*:

        - string, electron configuration of neutral atom
    
    ***
    '''
    try:
        return linecache.getline(filename,int(Z)).split(',')[1].strip()
    except IndexError:
        print('ERROR: Invalid atomic number, returning  nothing')
        return ''

      
def shield_split(shield_string):
    
    '''

    Parse the electron configuration string, dividing up into the different orbital
    shells.
    
    *args*:

        - **shield_string**: string, electron configuration
        
    *return*:

        - **parsed_shield**: list of separated orbital components
    
    ***
    '''
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
    
    '''
    Compute the effective nuclear charge, following Slater's rules.
    
    *args*:

        - **Z_ind**: int, the atomic number
        
        - **orb**: orbital string, written as nl in either fully numeric, or numeric-alpha format
    
    *return*:

        - **result**: float, effective nuclear charge
    
    ***
    '''
    
    l_dic = {"s":0,"p":1,"d":2,"f":3}  
    l_dic_inv = {0:"s",1:"p",2:"d",3:"f"}
    e_conf = get_con(filename,Z_ind)
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
    try:
        parsed_shield=shield_split(shield)
    except UnboundLocalError:
        print('ERROR: Invalid orbital combination given for Z = {:d}, returning None'.format(Z_ind))
        return None
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
    

def Slater_exec(Z_ind,orb):

    '''

    Define an executable Slater-type orbital wavefunction which takes only
    the radius as an input argument. In this way, the usser specifies Z 
    and the orbital label string, and this generates a lambda function
    
    :math:`R(r) = (\\frac{2Z_{e}}{n_{e}})^{n_{e}} \\sqrt{\\frac{2Z_{e}}{n_{e}\\Gamma(2n_{e}+1)}} r^{n_{e}-1} e^{-\\frac{Z_{e} r}{n_{e}}}`    
 
    *args*:

        - **Z_ind**: int, atomic number
        
        - **orb**: string, 'nlxx' as per standard orbital definitions used
        throughout the library
    
    *return*:

        - executable function of position (in units of Angstrom)
    
    ***
    '''
    ne = ndic[int(orb[0])]
    xi = Z_eff(Z_ind,orb)/ne
    tmp = (2*xi)**ne*np.sqrt(2*xi/gamma(float(2*ne)+1))
    
    def lambda_gen():

        return lambda r: tmp*(r)**(ne-1)*np.exp(-xi*r)
    return lambda_gen()
  
    


def hydrogenic_exec(Z_ind,orb):
    
    '''
    
    Similar to Slater_exec, we define an executable function for the 
    hydrogenic orbital related to the orbital defined with atomic number
    **Z_ind** and orbital label 'nlxx'
    
    *args*:

        - **Z_ind**: int, atomic number
        
        - **orb**: string, orbital label 'nlxx'
        
    *return*:
    
        - executable function of float
        
    ***
    '''
    
    mn = am.get_mass_from_number(Z_ind)
    au = (4*np.pi*Eo*hb**2*(mn*mp+me))/(q**2*mn*mp*me)
    n = int(orb[0])
    l = int(orb[1])
    
    def lambda_gen():
        return lambda r: (np.sqrt((2*Z_ind*A/(n*au))**3*factorial(n-l-1)/(2*n*factorial(n+l)))*np.exp(-Z_ind*r*A/(n*au))*(2*Z_ind*r*A/(n*au))**l*sc.genlaguerre(n-l-1,l*2+1)(2*Z_ind*r*A/(n*au)))
    
    return lambda_gen()

if __name__=="__main__":
    mn = am.get_mass_from_number(6)
    r = np.linspace(0,10,1000)
    Chyd = hydrogenic_exec(6,'21z')(r)
    CSlat= Slater_exec(6,'21x')(r)
    plt.figure()
    plt.plot(r,Chyd)
    plt.plot(r,CSlat)
    
