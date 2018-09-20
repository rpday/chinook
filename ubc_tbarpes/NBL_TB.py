# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 12:45:36 2018

@author: rday
"""

ao = 0.529177

import numpy as np
from operator import itemgetter


def load_params(fnm):
    
    '''
    Parameters recorded, ordered as a1,a2,a3,a4,B1,B2,d1,d2,d3 from the paper
    Each row labelled by the relevant potential e.g. Vsss, or PHI
    arg:filename string, origin of file
    return: dictionary of parameters, with keys representing the relevant potential, and the values the array of parameters
    '''
    order = ('a1','a2','a3','a4','b1','b2','d1','d2','d3')
    par_dict = {}
    with open(fnm,'r') as fromfile:
        for line in fromfile:
            tmp_line = line.split()
            if tmp_line[0]!='cx':
                par_dict[tmp_line[0]] = {order[o]:float(tmp_line[1+o]) for o in range(len(tmp_line)-1)}
            else:
                cut = float(tmp_line[1])
                npars = int(len(tmp_line[2:])/2)
                cpars = {'c{:d}'.format(ii-2):(float(tmp_line[ii]),float(tmp_line[ii+npars])) for ii in range(2,npars+2)}
                cpars.update({'cut':cut})
    return par_dict,cpars

def region(avec,num):
    '''
    Generate a symmetric grid of points in number of lattice vectors. The tacit assumption is a 3 dimensional lattice
    args: num -- integer--grid will have size 2*num+1 in each direction
    returns numpy array of size ((2*num+1)**3,3) with centre value of first entry of (-num,-num,-num),...,(0,0,0),...,(num,num,num)
    '''
    num_symm = 2*num+1
    return np.dot(np.array([[int(i/num_symm**2)-num,int(i/num_symm)%num_symm-num,i%num_symm-num] for i in range((num_symm)**3)]),avec)


def range_cutoff(R_cut,w):
    '''
    Define lambda function to give suppression of points beyond the R_cut, l_cut indicates how steeply this should
    cut-off
    
    R_cut,l_cut: floats, cutoff distance and rate
    return: function of float
    '''
    return lambda R: (1+np.exp((R-R_cut)/w))**(-1)


def xi(points,i,j,B):
    mpt = (points[i]+points[j])/2
    mid_approx = points[np.where(np.linalg.norm(points-mpt)==np.linalg.norm(points-mpt).min())[0][0]]
    pk = points + mid_approx 
    pk_mask = np.array([True if (np.linalg.norm(pki-points[i])>0.1 and np.linalg.norm(pki-points[j])>0.1) else False for pki in pk])
    pk = pk[pk_mask]
    rij = np.linalg.norm(points[j]-points[i])
    bond_terms = rij/(np.linalg.norm(pk-points[i],axis=1)+np.linalg.norm(pk-points[j],axis=1))
    xi_ij = np.sum(bond_terms*np.exp(-B[0]/(bond_terms)**B[1]))
    return xi_ij


def alt_xi(points,i,j,B):
    mpt = (points[i]+points[j])/2
    mid_approx = points[np.where(np.linalg.norm(points-mpt)==np.linalg.norm(points-mpt).min())[0][0]]
    pk = points + mid_approx 
    pk_mask = np.array([True if (np.linalg.norm(pki-points[i])>0.1 and np.linalg.norm(pki-points[j])>0.1) else False for pki in pk])
    pk = pk[pk_mask]
    rij = np.linalg.norm(points[j]-points[i])
    bond_terms = rij*(1/(np.linalg.norm(pk-points[i],axis=1))+1/(np.linalg.norm(pk-points[j],axis=1)))
    d = np.linalg.norm(0.5*(2*pk-points[i]-points[j]))**2
    xi_ij = np.sum(bond_terms*np.exp(-B[0]*d**B[1]))
    return xi_ij

def delta_ij(points,i,j,B):
#    return 2/np.pi*np.arctan(xi(points,i,j,B))# Cu
    return 2/np.pi*np.arctan(alt_xi(points,i,j,B)) #Ge
    

def Rij(points,i,j,pars):
    D = delta_ij(points,i,j,(pars['b1'],pars['b2']))
    return np.linalg.norm(points[j]-points[i])*(1+pars['d1']*D+pars['d2']*D**2+pars['d3']*D**3)

def SK_ij(points,i,j,Fc,pars):
    Rbond = Rij(points,i,j,pars)
    post = Rbond**(pars['a2'])*np.exp(-pars['a3']*Rbond**(pars['a4']))*Fc(Rbond)


    return pars['a1']*post


def SK_pars_gen(fnm,avec,Rc,w):
    
    pts = region(avec,2)
    Fc = range_cutoff(Rc,w)
    pars,_ = load_params(fnm)
    
    SK = {}
    pi = int(len(pts)/2)
    for pj in list(enumerate(pts)):
        if np.linalg.norm(pj[1])>0:
            lbl = '{:0.04f}'.format(np.linalg.norm(pj[1]))
            SK[lbl] = {}
            for par_ij in pars:
                SK[lbl][par_ij] = SK_ij(pts,pi,pj[0],Fc,pars[par_ij])
                
    return SK


def SK_r(fnm,avec,Rc,w):
    
    pars,_ = load_params(fnm)
    Fc = range_cutoff(Rc,w)
    avec_scale = avec/np.linalg.norm(avec,axis=1)
    SK_f = {}
    r = np.linspace(1.5,6.5,10)
    for si in pars:
        SK_f[si] = np.zeros(10)
        for ii in range(10):
            pts = region(r[ii]*avec_scale,1)
            SK_f[si][ii] = SK_ij(pts,13,14,Fc,pars[si])
    return SK_f

def pair_pot(fnm,avec):
    Rc,w=6.0,0.1 #CU
#    Rc,w=6.0,0.05 #Ge
    pars,cpars = load_params(fnm)
    fc = lam_fpoly(cpars)
    Fc = range_cutoff(Rc,w)
    pts = region(avec,2)
    phi_vals = []
    for ii in range(len(pts)):    
        if ii!=int(len(pts)/2):
            phi_vals.append(SK_ij(pts,int(len(pts)/2),ii,Fc,pars['phi']))
    phi_vals = np.array(phi_vals)
    fcr = fc(np.array([np.sum(phi_vals)]))
    return fcr
        
    
def lam_fpoly(cpars):
    
    return lambda x: f_poly(x,cpars)
    
def f_poly(x,cpars):
    
    cut = cpars['cut']
    out = np.zeros(len(x))
    out[:(len(x[x<=cut]))] = poly(x[x<=cut],[cpars['c{:d}'.format(ii)][0] for ii in range(len(cpars)-1)])
    out[(len(x[x<=cut])):] = poly(x[x>cut],[cpars['c{:d}'.format(ii)][1] for ii in range(len(cpars)-1)])
    return out

def poly(x,c):
    if len(c)<1:
        return 0
    else:
        return x**(len(c)-1)*c[-1] + poly(x,c[:-1])

def recondition(SK,onsite,tol):
    tdict = {'ssr':'004400S','spr':'004301S','sdr':'004302S',
             'ppr':'003311S','ppp':'003311P','pdr':'003312S',
             'pdp':'003312P','ddr':'003322S','ddp':'003322P',
             'ddd':'003322D','phi':'phi'} #CU
#    tdict = {'ssr':'004400S','spr':'004401S','ppr':'004411S',
#             'ppp':'004411P','sSr':'004500S','pSr':'004510S',
#             'SSr':'005500S','phi':'phi'} #GE
    SKout = []
    dists = np.array([float(si) for si in SK])
    
    cutoff = []
    for si in SK:
        tmp_dict = SK[si]
        keys = [tdict[si_k] for si_k in tmp_dict if si_k in tdict]
        vals = np.array([tmp_dict[si_k] for si_k in tmp_dict ])
        if abs(vals).max()<tol:
            continue
        else:
            SK_tmp = {keys[i]:vals[i] for i in range(len(vals))}
            if float(si)==abs(dists).min():
                SK_tmp.update(onsite)
            SKout.append(SK_tmp)
            cutoff.append(float(si)*1.001)
    SKout,cutoff= sort_SK(SKout,cutoff)
#    print(dists)
#    print(cutoff)
    return SKout,cutoff

def sort_SK(SK,C):

    cuts = [[i,C[i]] for i in range(len(C))]
    cuts_sort = np.array(sorted(cuts,key=itemgetter(1)))
    SKp = [SK[int(cuts_sort[i,0])] for i in range(len(cuts))]
    return SKp,list(cuts_sort[:,1])
        


def gen_Cu_SK(avec,fnm,tol):
    Rc,w=6.0,0.1#CU
#    Rc,w = 6.0,0.05 #Ge
    onsite = {"040":-2.408,"031":4.00,"032":-5.00} #CU
#    onsite = {"040":-6.50,'041':2.15,'050':10.35} #GE
    SK = SK_pars_gen(fnm,avec,Rc,w)
    TB_SK,CUT = recondition(SK,onsite,tol)
    return TB_SK,CUT
        
    
    




if __name__ == "__main__":
    
    a = 3.606
    avec = np.array([[a/2,a/2,0],[0,a/2,a/2],[a/2,0,a/2]])
    
    Rc,w = 6.0,0.05
    Fc = range_cutoff(Rc,w)
#    fnm = 'C:/Users/rday/Documents/TB_ARPES/2018/TB_ARPES_2018/TB_ARPES-master/examples/cu_params_pan.txt'
    fnm = 'C:/Users/rday/Documents/TB_ARPES/2018/TB_ARPES_2018/TB_ARPES-master/examples/germanium_params.txt'

#    SK = SK_pars_gen(fnm,avec,Rc,w)
    SK,C = gen_Cu_SK(avec,fnm,0.001)
    SKGe = SK_r(fnm,avec,Rc,w)
#    Er = pair_pot(fnm,avec,Rc,w)
    
#    pars.update({'s':-2.408,'p':4.0,'d':-5.0})
    
    
    