#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 20:10:05 2017

@author: ryanday
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sc

class pp_wf:
    
    def __init__(self,pars,values,rab,r,mesh):
        self.pars = pars_parse(pars)
        self.mesh = pars_parse(mesh) #can be used to determine the form or r(i)
        self.values = np.array(values)
        self.rab = np.array(rab) #rab(mesh): factor required for discrete integration: ∫ f(r) dr =∑ifi rabi
        self.r = np.array(r) #r(i) = e(xmin+i*dx)/Zmesh or or r(i) = (e(xmin+i*dx)-1)/Zmesh
        
 
    

    def pseudo_integral(self,kn):
        '''
        Calculate the radial integrals for a pseudowavefunction from file.
        Args:
            filename -- string of origin of the PP file
            orb -- string: the orbital of interest-- format "nlxxx" with xxx the cubic channel, eg xy
            kn -- numpy array of float of the k vector lengths of interest
        Returns:
            Bm,Bp -- numpy array
        
        '''
        
        Bm = np.zeros(len(kn),dtype=complex)
        Bp = np.zeros(len(kn),dtype=complex)
        
        ldic = {"S":0,"P":1,"D":2,"F":3}
        
        lp = [ldic[self.pars['label'][-1]]-1,ldic[self.pars['label'][-1]]+1]
        
        for i in range(len(kn)):
            jm = sc.spherical_jn(lp[0],kn[i]*self.r*0.529177) #the pseudopotential in units of Bohr
            jp = sc.spherical_jn(lp[1],kn[i]*self.r*0.529177)
            
            Bm[i] = (-1.0j)**lp[0]*sum(self.r**2*self.values*jm*self.rab) #PP functions are actually r*Chi(r) so r^3 need only be written as r^2
            Bp[i] = (-1.0j)**lp[1]*sum(self.r**2*self.values*jp*self.rab)
            
            '''
            Next step is to load the PP file into an attribute of the orbitals in the Lattice.
            Then when running a photoemission experiment, run this pseudo_integral function with a coarse k mesh
            and then finally generate an interpolation grid for then finding the exact values of B for the k values of
            interest
            Note that we output all l' channels here so need to make a modified Bdic_Gen function which will
            run this, interpol
            '''
            
        return Bm,Bp
    
                


def parse_pp(filenm):
    reading = False
    
    PP_count = 1
    PPs = {}
    vals = []
    rab = [] #spacing for discrete integration
    r = [] #r mesh spacing
    rab_coll = False
    r_coll = False
    with open(filenm,'r') as pp_file:
        for l in pp_file:
            
            if "PP_MESH dx" in l:
                l2= next(pp_file)
                mesh = l[:-1]+" "+l2[:-2]
                
            
            if "PP_RAB " in l:
                rab_coll = True
            if rab_coll:
                if "/PP_RAB" in l:
                    rab_coll = False
                else:
                    try:
                        tmp = float(l.split()[0])
                        rab += [float(li) for li in l.split()]
                    except ValueError:
                        continue
            if "PP_R " in l:
                r_coll = True
            if r_coll:
                if "/PP_R" in l:
                    r_coll = False
                else:
                    try:
                        tmp = float(l.split()[0])
                        r+=[float(li) for li in l.split()]
                    except ValueError:
                        continue
                    
            if "PP_AEWFC.{:d}".format(PP_count) in l:
                if len(l.split())>3:
#                    l2 = next(pp_file)
                    reading = True
                    pars = l[:-1] #+ l2[:-1]
                else:
                    reading = False
                    tmp_PP = pp_wf(pars,vals,rab,r,mesh)
                    norm_PP = np.around(sum(tmp_PP.values**2*tmp_PP.rab))

                    
                    PPs[PP_count] =tmp_PP
                    PP_count+=1
                  #  PP_count+=1

                    vals = []
                    
            if reading:
                try:
                    tmp = float(l.split()[0])
                    vals+=[float(li) for li in l.split()]
                except ValueError:
                    continue
                    

            if "<PP_PSWFC" in l:
                reading = False
    pp_file.close()
    return PPs

def pars_parse(pars):
        pars_dict = {}
        psplit = pars.split(' ')
        for p in psplit:
            tmp = p.split('=')
            if len(tmp)==2:
                try:
                    val = float(tmp[1][1:-1])
                except ValueError:
                    val = tmp[1][1:-1]
                pars_dict[tmp[0]] = val
        return pars_dict
    
    
if __name__=="__main__":
    filenm = "C:/Users/rday/Documents/TB_ARPES/2017/August/As.txt.txt"
    PPs = parse_pp(filenm)  
      
    Eb = np.arange(-0.45,0.2,0.025)
    hv = np.linspace(6,100,200)
    kn = np.sqrt(2*9.11*10**-31/(6.626*10**-34/(2*np.pi))**2*1.602*10**-19*(hv-4.45+0.0))/10**10
    Bm,Bp = PPs[3].pseudo_integral(kn)
    plt.figure()
    plt.plot(hv,Bm)
    plt.plot(hv,Bp)
    ratio = Bm/Bp
    fig = plt.figure()
    ax = fig.add_subplot(111)
    p = ax.plot(hv,abs(ratio))
    
#    leglist =[]
#    fig = plt.figure()
#    for i in range(1,len(PPs)+1):
#        norm_PP = np.around(sum(PPs[i].values**2*PPs[i].rab))
#        if norm_PP==1.0:
#            leglist.append(PPs[i].pars['label'])
#            plt.plot(PPs[i].r/0.529177,PPs[i].values/PPs[i].r,lw=2)
#    plt.axis([0,20,-3,3])
#    plt.legend(leglist)
#    plt.savefig('As.pdf',format='pdf')
#    
    
                
            
           