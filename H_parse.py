# -*- coding: utf-8 -*-
"""
Created on Thu Dec 08 10:45:13 2016

@author: rday
"""

import numpy as np
import re


def wout_parse(lenbase,woutfile):
    pos = []
    a_vec = np.zeros((3,3))
    with open(woutfile,"r") as wout:
        for line in wout:
            tmp = line.split()
            if len(tmp)!=0:
                if tmp[0]=="a_1":
                    a_vec[0,:] = np.array([float(tmp[1]),float(tmp[2]),float(tmp[3])])
                    print("a1: ",a_vec[0,:])
                if tmp[0]=="a_2":
                    a_vec[1,:] = np.array([float(tmp[1]),float(tmp[2]),float(tmp[3])])
                    print("a2: ",a_vec[1,:])
                if tmp[0]=="a_3":
                    a_vec[2,:] = np.array([float(tmp[1]),float(tmp[2]),float(tmp[3])])
                    print("a3: ",a_vec[2,:])
            if len(tmp)>4:
                if tmp[0]+tmp[1]+tmp[2]+tmp[3] =="WFcentreandspread":
                    tmp2 = [re.sub(',','',t) for t in tmp[6:9]]
                    pos.append([float(tmp2[0]),float(tmp2[1]),float(tmp2[2])])
    wout.close()
    return a_vec,np.array(pos[-lenbase:])


def read_H(fn):
    tmp = []
    with open(fn,"r") as readfile:
        lc = 0  
        Hr = False
        Hr_start = 1000 #initialize Hr_start value, to be redefined in an if statement
        for line in readfile:
            if lc==2:
                n_WS = int(line.split()[-1])
                Hr_start = int(np.ceil(n_WS/15)+4) #Wannier90 prints out the degeneracy of each Wigner Seitz point in lines of 15
            if lc==Hr_start:
                Hr = True
            if Hr:
                lineparse = [float(st) for st in line.split()] #Only the latter part of the data file is of interest
                tmp.append(lineparse)
            lc+=1
    readfile.close()  
    return tmp
    
def trun_H(Hlist,tol):
    tmp = [h for h in Hlist if abs(h[-2]+1.0j*h[-1])>tol]
    return tmp

def base_change(Hlist,base_v):
    S = base_v*np.identity(len(base_v),dtype=complex)
    Si = np.linalg.inv(S)
    Hnew = []
    for h in Hlist:
        Htmp = np.zeros((len(base_v),len(base_v)),dtype=complex)
        Htmp[int(h[3])-1,int(h[4])-1] = h[5]+1.0j*h[6]
        Hmat = np.dot(Si,Htmp*base_v)
        new_els = np.transpose(np.nonzero(Hmat))
        for n in new_els:
            matel = Hmat[n[0],n[1]]
            Hnew.append([h[0],h[1],h[2],n[0]+1,n[1]+1,np.real(matel),np.imag(matel)])
    return Hnew

def clean_H(Hlist,pos,avec,double):
    tmp = [h for h in Hlist]
    Hclean = []
    print(tmp[0])
    print(tmp[1])
    for h in tmp:
        Rij = avec[0]*h[0]+avec[1]*h[1]+avec[2]*h[2]-pos[int(h[3]-1)]+pos[int(h[4]-1)]
        Hval = str(complex(np.around(h[5],5)+np.around(h[6],5)*1.0j))
        
        #tmp2 = str(h[4]-1)+','+str(h[5]-1)+','+str(Rij[0])+','+str(Rij[1])+','+str(Rij[2])+','+
        tmp2 = '%d,%d,%0.5f,%0.5f,%0.5f,%s' % (h[3]-1,h[4]-1,Rij[0],Rij[1],Rij[2],Hval)
        Hclean.append(tmp2)
        if double:
            tmp3 = '%d,%d,%0.5f,%0.5f,%0.5f,%s' % (h[3]-1+len(pos),h[4]-1+len(pos),Rij[0],Rij[1],Rij[2],Hval)
            Hclean.append(tmp3)
    return np.array(Hclean)

def write_H(Hlist,fn):
    with open(fn,"w") as writefile:
        for Hij_line in Hlist:
            wl = Hij_line+",\n"
            writefile.write(wl)
        writefile.close()
    print('Hamiltonian written to file')
    return True
    
if __name__=="__main__":
    
    ###file origin and file destination
    filename = "fese_hr.dat" #"C:\\Users\\rday\\Documents\\TB_ARPES\\FeSe\\fese_hr.dat"
    savename = "C:\\Users\\rday\\Documents\\TB_ARPES\\2018\\TB_ARPES_2018\\fese_march.txt" #"C:\\Users\\rday\\Documents\\TB_ARPES\\FeSe\\FeSe_espresso_HAMR.txt"
    
    ###min value for non-zero hopping matrix element
    trunc_tol = 5*10**-10
    
    ###number of orbitals in the basis
    lenbase = 16
    bohr = 0.529177249 #bohr to Angstrom
    a,c=7.1255903*bohr,10.433178*bohr
#    avecs = np.array([[3.7734,0,0],[0,3.7734,0],[0,0,5.5258]]) #FeSe
    avecs = np.array([[a,0,0],[0,a,0,],[0,0,c]])

#    olist = ['2xy','2xz','2yz','2XY','2ZR','2xy','2xz','2yz','2XY','2ZR','1x','1y','1z','1x','1y','1z']
    olist = ['2ZR','2xz','2yz','2XY','2xy','2ZR','2xz','2yz','2XY','2xy','2pz','2px','2py','2pz','2px','2py']
   ###pos --array of atomic positions
#    pos = np.array([avecs[0]*0.75+0.25*avecs[1],0.25*avecs[0]+0.75*avecs[1],0.25*avecs[0]+0.25*avecs[1]+0.2672*avecs[2],0.75*avecs[0]+0.75*avecs[1]+0.7328*avecs[2]]) # FeSe
    pos = np.array([avecs[0]*0.75+0.25*avecs[1],0.25*avecs[0]+0.75*avecs[1],0.25*avecs[0]+0.25*avecs[1]+0.26668*avecs[2],0.75*avecs[0]+0.75*avecs[1]+0.73332*avecs[2]]) # LiFeAs
   ###for generation of position array with position for each atom in basis
    l_list = np.array([2,2,1,1])
    full_pos = np.array([pos[x] for x in range(4) for m in range(2*l_list[x]+1)])
    
    ##read in the Hamr file
    Hamlist = read_H(filename)
    
    ###clean the Hamr file
    truncate = trun_H(Hamlist,trunc_tol)
    
    ##transform the basis as necessary:
 ###wannier sometimes rotates the orbitals from the canonical definitions--check and change phase as necessary
#    base_v = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],dtype=complex) #Wannier90 giving xy and yz defined with negative sign overall
#    trans_HAM = base_change(truncate,base_v)
    
#    print 'original file length: %d'%len(Hamlist)
#    print 'truncated and transformed for Hij>10^-4: %d'%len(trans_HAM)
    
    ##reformat the output above for writing to file
    Hij = clean_H(truncate,full_pos,avecs,double=False)
    
#    Hij = clean_H(trans_HAM,full_pos,avecs)
    write_H(Hij,savename)

    