#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 09:55:47 2018

@author: ryanday

build a UBC_TBARPES.py input file, taking the Wannier files as input

"""
import sys
sys.path.append('C:/Users/rday/Documents/TB_ARPES/2018/TB_ARPES_2018/TB_ARPES-master/')
import ubc_tbarpes.wannier_load as w90
import numpy as np



def generate_py(seedname):
    hr = seedname+'_hr.dat'
    inwf = seedname + '.inwf'
    win = seedname + '.win'
    pyname = seedname + '_wann.py'
    hamtext = seedname + '_wann.txt'
    wannmodel = w90.wannier(hr,win,inwf)
    w90.write_ham(wannmodel.hamiltonian,hamtext)
    return wannmodel,pyname,hamtext
    
    
    
def import_libraries(home_path):
    write = 'import sys\n'
    write+='sys.path.append("{:s}")\n\n'.format(home_path)
    write+='import numpy as np\n'
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


def basis_dict(wannmodel,Z,n):
    orb_list = gen_orblist(wannmodel.basis,n)
    write = 'basis_dict = {"atoms":['+','.join('{:d}'.format(i) for i in range(len(wannmodel.pos)))+'],\n'
    write+='\t "Z":{'+','.join('{:d}'.format(i)+':{:d}'.format(Z[i]) for i in range(len(Z)))+'},\n'
    write+='\t "orbs":['+','.join('['+','.join('"{:s}"'.format(orb_list[i][j]) for j in range(len(orb_list[i])))+']' for i in range(len(orb_list)))+'],\n'
    write+='\t "pos":['+','.join('np.array([{:0.04f},{:0.04f},{:0.04f}])'.format(*wannmodel.pos[i]) for i in range(len(wannmodel.pos)))+'],\n'
    write+='\t"spin":spin_dict}\n\n'
    return write

def k_dict(wannmodel):
    
    write = 'K_dict = {"type":"F",\n'
    write+= '\t"avec":avec,\n'
    write+='\t"pts":[np.array([0,0,0]),np.array([0,0.5,0.0]),np.array([0,0.5,0.5])],\n'
    write+='\t"grain":200,\n'
    write+='\t"labels":["$\Gamma$","M"]}\n\n'
    return write

def ham_dict(wannmodel,hamfile):
    tol = (abs(np.array([h[-1] for h in wannmodel.hamiltonian]))>0).min()*0.1
    cutoff = abs(np.array([np.sqrt(h[2]**2+h[3]**2+h[4]**2) for h in wannmodel.hamiltonian])).max()*1.01
    ren = 1.0
    off = 0.0
    write = 'H_dict = {"type":"txt",\n'
    write+= '\t"filename":"{:s}",\n'.format(hamfile)
    write+='\t"cutoff":{:0.04f},\n'.format(cutoff)
    write+='\t"renorm":{:0.04f},\n'.format(ren)
    write+='\t"offset":{:0.04f},\n'.format(off)
    write+='\t"tol":{:0.04f},\n'.format(tol)
    write+='\t"spin":spin_dict,\n'
    write+='\t"avec":avec}\n\n'
    return write


def build():
    write = 'basis_dict = build_lib.gen_basis(basis_dict)\n'
    write+= 'Kobj = build_lib.gen_K(K_dict)\n'
    write+= 'TB = build_lib.gen_TB(basis_dict,H_dict,Kobj)\n\n'
    write+= 'TB.solve_H()\n'
    write+= 'TB.plotting()\n\n'
    return write
    
    
    
    
  
    
def write_file(homepath,seedname,Z,n):
    wmodel,pyfile,hamfile = generate_py(seedname)
    with open(pyfile,'w') as destn:
        destn.write(import_libraries(homepath))
        destn.write(lattice(wmodel.avec))
        destn.write(spin_dict(len(wmodel.pos)))
        destn.write(basis_dict(wmodel,Z,n))
        destn.write(k_dict(wmodel))
        destn.write(ham_dict(wmodel,hamfile))
        destn.write(build())    
    
    
def gen_orblist(basis,n):
    orbs =[]
    for o in basis:
        label = '{:d}{:s}'.format(n[len(orbs)],o.label)
        if (o.atom+1)==len(orbs):
            orbs[-1].append(label)
        else:
            orbs.append([label])
    return orbs
            
    
    
if __name__ == "__main__":
    
#    seedname = '/Users/ryanday/Documents/UBC/TB_ARPES/TB_ARPES-master 02102018/examples/Bi2Se3/Bi2Se3'
#    homepath = '/Users/ryanday/Documents/UBC/TB_ARPES/TB_ARPES-master 02102018/'
    seedname = 'C:/Users/rday/Documents/TB_ARPES/2018/TB_ARPES_2018/TB_ARPES-master/examples/Bi2Se3/Bi2Se3'
    homepath = 'C:/Users/rday/Documents/TB_ARPES/2018/TB_ARPES_2018/TB_ARPES-master/'
    Z = [83,83,34,34,34,34]
    n = [6,6,4,4,4,4]
    write_file(homepath,seedname,Z,n)