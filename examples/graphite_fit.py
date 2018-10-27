# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 22:17:12 2018

@author: rday
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import ubc_tbarpes.build_lib as blib
import graphite_shell as graphite

###---------------------    LOAD DATASET   ---------------------------------###
    
def load_data(filename):
    with open(filename,'r') as fromfile:
        data = []        
        for line in fromfile:
            try:
                tmp = [float(t) for t in line.split(',')]
                if len(tmp)>0:
                    data.append(tmp)

            except ValueError:
                continue
    return np.array(data)

def local_mean(arr,start,bins):
    '''
    Quick smoothing of the dataset to find the local average within a region. Deal
    with edges by using PBC. Fine for the parabolic dataset used here, for other
    BC, would probably need to just use the same average when we are near the edges.
    E.g. for monotonic set, using PBC would mess up the points near end.
    
    args:
        arr: data array shape (N) float
        start: start index (int) in arr
        bins: length of range to sum-over
    '''
    return 1./bins*np.sum([arr[np.mod(-int(bins/2)+start+ii,len(arr))] for ii in range(bins)])

def separate_bands(data):
    '''
    Dataset sent as k,E in two columns, but not clearly split into two well-defined bands.
    It's not as simple as interwoven, they are spliced into one another, and no apparent order.
    To deal with this, locally average, here 20 points, taking PBC on the array, if the point is
    higher than the local average, it's taken as the upper band, and else, lower band. Check and 
    it works, easily separates the two bands. Great! 
    Note: think about how to possibly generalize such a routine to have more than 2 bands...
    args:
        data: numpy array of float shape(N,2)
    return:
        b1,b2: numpy arrays of float, shape (N-len(b1),2), (N-len(b2),2)
    '''
    b1,b2 = [],[]
    bins = 20
    for i in range(len(data)):
        if data[i,1]>=local_mean(data[:,1],i,bins):
            b2.append(data[i])
        else:
            b1.append(data[i])
    return np.array(b1),np.array(b2)

def fit_dataset(b1,b2):
    tmp = np.zeros(((len(b1)+len(b2)),2))
    tmp[:len(b1)] = b1
    tmp[len(b1):] = b2
    return tmp

###---------------------    LOAD DATASET   ---------------------------------###
###############################################################################
###############################################################################
###----------------------  PREPARE MODEL --- -------------------------------###
def build_K(b1,b2,kz):
    '''
    Very specialized function, for graphite, looking at GK direction, so along ky direction
    args:
        b1,b2: data-arrays containing k and E for the 2 bands, numpy array of (N,2), (M,2) float
        kz: float kz value assumed
    return:
        instance of Kobj class
    '''
    klist = [np.array([0,ki,kz]) for ki in b1[:,0]] + [np.array([0,ki,kz]) for ki in b2[:,0]]
    Kd = {'type':'A',
			'pts':klist,
			'grain':1,
			'labels':[]}
    Kobj = blib.gen_K(Kd)
    return Kobj

def lin_K(kmin,kmax,Nk,kz):
    '''
    Very quickly, for the plotting of output, make a new k-path with regularly spaced points
    to plot in comparison with data
    args:
        kmin: initial k-value (along ky) float
        kmax: final ''
        Nk: number of k-points
        kz: float kz value
    return:
        Kobj instance
    '''
    Kd = {'type':'A',
			'pts':[np.array([0,kmin,kz]),np.array([0,kmax,kz])],
			'grain':Nk,
			'labels':[]}
    Kobj = blib.gen_K(Kd)
    return Kobj
    

def build_TB(Kobj):
    '''
    wrapper for build_TB in the graphite_shell script
    args:
        Kobj: instance of Kobj class
    return:
        TB: instance of TB class
    '''
    return graphite.build_TB(Kobj)


def precondition_H(mats):
    '''
    Standard format of the TB.mat_els.H is a list, with the last element complex. 
    For this case, we have only real hoppings, and so we can transform the entire thing into an array of float.
    Do this here
    '''
    for hi in range(len(mats)):
        for hij in range(len(mats[hi].H)):
            mats[hi].H[hij][-1] = float(np.real(mats[hi].H[hij][-1]))
        mats[hi].H = np.array(mats[hi].H)
    return mats

    

###----------------------  PREPARE MODEL --- -------------------------------###
###############################################################################
###############################################################################
###-------------------    FITTING PROCEDURE -------------------------------###

def rebuild_H(args,mats):
    '''
    Rebuild the Hamiltonian with modified TB parameters
    '''
    tdic = [[5,7,6,5],[0,0,0],[1,1],[4,4,4,4,4,4],[2,7,2],[4,4,4,4,4,4],[3,3,3,3,3,3],[5,7,6,5],[0,0,0],[2,7,2]]
    print('TBARGS',args,'\n')
    for i in range(len(mats)):
        tmp = mats[i].H
        tmp[:,-1] = np.array(args)[tdic[i]]
        mats[i].H = tmp
    return mats

def fit_func(kpts,*TBargs):
    '''
    Redefine the tight-binding matrix elements, then solve the Hamiltonian and splice the two bands of interest into a
    single array to return as the function values
    '''
    TB.mat_els = rebuild_H(TBargs,TB.mat_els)
    TB.solve_H()
    y= np.zeros(len(kpts))
    y[:len(b1)] = TB.Eband[:len(b1),0]
    y[len(b1):] = TB.Eband[len(b1):,1]
    return y


def fit_TB(kpts,energies,tog,b_min,b_max):
    '''
    Perform the fit!
    kpts: numpy array of N floats corresponding to k-values from MDC fit
    energies: numpy array of N floats corresponding to MDC peak energie values
    tog: initial guess of the tight-binding parameters
    b_min: lower  bound on fit parameters (list of float)
    b_max: upper bound on fit parameters (list of float)
    '''
    c,co = curve_fit(fit_func,kpts,energies,p0=tog,bounds =(b_min,b_max))
    return c,co
    
###---------------------    FITTING PROCEDURE -------------------------------###
###############################################################################

def fit_summary(c,tog,bands,TB,kmin,kmax,kz):
    '''
    Basic-- print out some fit results, and then plot the fitted tight-binding
    model alongside the data for comparison
    args:
        c: curve_fit output parameters (numpy array of 8 float)
        tog: initial parameters (tuple of 8 float)
        bands: dataset numpy array of (N,2) float
        TB: Tight-binding model
        kmin,kmax: float endpoints for calculated fit curves
        kz: float kz value
    return:
        None
    '''
    tnm = ('t0','t1','t2','t3','t4','t5','D','E')
    print(' Par |  Initial      Final')
    print('##########################')
    for i in range(len(tog)):
        print(' {:4}|  {:0.4f}    {:0.4f}  '.format(tnm[i],tog[i],c[i]))
    print('##########################')
    TB.mat_els = rebuild_H(c,TB.mat_els)
    TB.Kobj = lin_K(kmin,kmax,200,kz)
    TB.solve_H()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(bands[:,0],bands[:,1])
    for i in range(4):
      ax.plot(TB.Kobj.kpts[:,1],TB.Eband[:,i],c='k')  

    


if __name__ == "__main__":
    kz = 2.7 ##set kz value
    tog = (-3.12,0.355,-0.010,0.24,0.12,0.019,-0.008,-0.024)   ##Initial Parameters
    dfile = 'Gruneis_MDC_fit.txt' ###Dataset TextFile
    darr = load_data(dfile) ###Load Datafile
    b1,b2 = separate_bands(darr) ##make data useable
    bands = fit_dataset(b1,b2) ### make data useable and 1D
    Kobj = build_K(b1,b2,kz) ## Build the k-path
    TB = build_TB(Kobj) ## build the TB model
    TB.mat_els = precondition_H(TB.mat_els) ###make TB matrix elements more functional for fitting procedure
    #### For reference, the parameters are organized as: ('t0','t1','t2','t3','t4','t5','D','E')###
    t_min = [-3.5,0.0,-0.5,0.1,0.05,0.0,-0.1,-0.5]##lower bounds on parameters
    t_max = [-3.,0.5,0.0,0.5,0.5,0.2,0.1,-0.015] ## upper bounds on parameters

    c, co= fit_TB(bands[:,0],bands[:,1],tog,t_min,t_max) ## perform the fitting
    
    kmin,kmax = 0,1.8
    fit_summary(c,tog,bands,TB,kmin,kmax,kz) ##compare data against fit result
    
    
    

    