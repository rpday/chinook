#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 09:55:47 2018

@author: ryanday

build a UBC_TBARPES.py input file, taking the Wannier files as input

"""
import ubc_tbarpes.wannier_load as w90



def generate_py(seedname):
    hr = seedname+'_hr.dat'
    inwf = seedname + '.inwf'
    win = seedname + '.win'
    pyname = seedname + '_wann.py'
    wannmodel = w90.wannier(hr,win,inwf)
    return wannmodel,pyname
    
    
    
def import_libraries(home_path):
    write = 'import numpy as np\nimport sys\n'
    write+='sys.path.append("{:s}")\n'.format(home_path)

    write+='import ubc_tbarpes.build_lib as build_lib\n'
    write+='import ubc_tbarpes.ARPES_lib as ARPES \n\n\n'
    return write


def lattice(vecs):
    avec = 'avec=np.array(['+','.join(['[{:0.04f},{:0.04f},{:0.04f}]'.format(*vecs[i]) for i in range(3)])+'])\n\n'
    return avec


def spin_dict(natoms):
    write = 'spin_dict = {"bool":False,\n'
    write+= '\t"soc":False,\n'
    write+='\t"lam":{'+','.join('{:d}:0.0'.format(i) for i in range(natoms))+'},\n'
    write+='\t"order":"N"}\n\n'
    return write


def basis_dict(wannmodel,Z):
    write = 'basis_dict = {"atoms":['+','.join('{:d}'.format(i) for i in range(len(wannmodel.pos)))+'],\n'
    write+='\t "Z":{'+','.join('{:d}'.format(i)+':{:d}'.format(Z[i]) for i in range(len(Z)))+'},\n'
#    write+='\t "orbs":'+orblist(wannmodel)+',\n'
    write+='\t "pos":['+','.join('np.array([{:0.04f},{:0.04f},{:0.04f}])'.format(*wannmodel.pos[i]) for i in range(len(wannmodel.pos)))+'],\n'
    write+='\t"spin":spin_dict}\n\n'
    return write

def k_dict(wannmodel):
    
    write = 'K_dict = {"type":"F",\n'
    write+= '"avec":avec,\n',
    write+='"pts":[np.array([0,0,0]),np.array([0,0.5,0.0]),np.array([0,0.5,0.5])],\n'
    write+='"grain":200,\n'
    write+='"labels":["$\Gamma$","M"]}\n\n'
    return write

def h_dict(wannmodel):
    

ham_dict = {'type':'txt',
			'filename':filenm,
			'cutoff':CUT,
			'renorm':REN,
			'offset':OFF,
			'tol':TOL,
			'spin':spin_dict,
         'avec':avec}
    
    
  
    
def write_file(homepath,seedname,Z):
    wmodel,pyfile = generate_py(seedname)
    with open(pyfile,'w') as destn:
        destn.write(import_libraries(homepath))
        destn.write(lattice(wmodel.avec))
        destn.write(spin_dict(len(wmodel.pos)))
        destn.write(basis_dict(wmodel,Z))
        destn.write(k_dict(wmodel))
        
    
    
    
if __name__ == "__main__":
    
    seedname = '/Users/ryanday/Documents/UBC/TB_ARPES/TB_ARPES-master 02102018/examples/Bi2Se3/Bi2Se3'
    homepath = '/Users/ryanday/Documents/UBC/TB_ARPES/TB_ARPES-master 02102018/'
    Z = [83,83,34,34,34,34]
    write_file(homepath,seedname,Z)