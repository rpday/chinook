# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 11:37:32 2018

@author: rday
"""
import numpy as np

a,c = 2.46,3.35
k_args = {'+kxl':np.array([a/np.sqrt(3.),0,0]),'-kxl':np.array([-a/np.sqrt(3),0,0]),
          '+kxs':np.array([a/(2*np.sqrt(3)),0,0]),'-kxs':np.array([-a/(2*np.sqrt(3)),0,0]),
          'ky':np.array([0,a/2,0]),'kz':np.array([0,0,c]),'k2z':np.array([0,0,2*c]),'-k2z':np.array([0,0,-2*c])}

klist = [ki for ki in k_args]





def cos(v,arg):
    tmp_v = []
    for vi in v:
        vec = np.array([vi[0],vi[1],vi[2]])
        tmp_v.append([*(vec+k_args[arg]),0.5*vi[-1]])
        tmp_v.append([*(vec-k_args[arg]),0.5*vi[-1]])
    return tmp_v

def sin(v,arg):
    tmp_v = []
    for vi in v:
        vec = np.array([vi[0],vi[1],vi[2]])
        tmp_v.append([*(vec+k_args[arg]),-0.5j*vi[-1]])
        tmp_v.append([*(vec-k_args[arg]),0.5j*vi[-1]])
    return tmp_v

def exp(v,arg):
    tmp_v = []
    for vi in v:
        vec = np.array([vi[0],vi[1],vi[2]])
        tmp_v.append([*(vec+k_args[arg]),vi[-1]])

    return tmp_v

def Hparse(Hstring,pars):



    
    
    tmp = Hstring.split(',')
    ii = int(tmp[0])-1
    jj = int(tmp[1])-1
    H_els = []
    for i in range(2,len(tmp)):
        init_H = [[0.0,0.0,0.0,1.0]]
        raw_str = tmp[i].split('*')
        for r_el in raw_str:
            try:
                init_H[0][-1] *= complex(r_el)
            except ValueError:
                True
            try:
                init_H[0][-1] *= pars[r_el]
            except KeyError:
                True
            if r_el[:3]=='cos':
                init_H = cos(init_H,r_el[4:-1])
            elif r_el[:3] == 'sin':
                init_H = sin(init_H,r_el[4:-1])
            elif r_el[:3] == 'exp':
                init_H = exp(init_H,r_el[4:-1])
        for hi in init_H:
            H_els.append([ii,jj,*hi])
    return H_els



def Hbuild(pars):
    Hlist  = ['1,1,E,D,2*t5*cos(k2z)',
      '1,2,t0*exp(+kxl),2*t0*cos(ky)*exp(-kxs)',
      '1,3,2*t1*cos(kz)',
    '1,4,2*t4*cos(kz)*exp(-kxl),4*t4*cos(kz)*cos(ky)*exp(+kxs)',
    '2,2,E,2*t2*cos(k2z)',
    '2,3,2*t4*cos(kz)*exp(-kxl),4*t4*cos(kz)*cos(ky)*exp(+kxs)',
    '2,4,2*t3*cos(kz)*exp(+kxl),4*t3*cos(kz)*cos(ky)*exp(-kxs)',
    '3,3,E,D,2*t5*cos(k2z)',
    '3,4,t0*exp(-kxl),2*t0*cos(ky)*exp(+kxs)',
    '4,4,E,2*t2*cos(k2z)']
    H_full = []
    for hi in Hlist:
        H_full += Hparse(hi,pars)
    return H_full

def Hwrite(Ham,filenm):
    
    with open(filenm,'w') as tofile:
        for hi in Ham:
            hij = '('+str(np.real(hi[-1]))+('+' if np.imag(hi[-1])>=0 else '-')+str(np.imag(hi[-1]))+'j)'
            tmpln = '{:d},{:d},{:0.6f},{:0.6f},{:0.6f},{:s}\n'.format(hi[0],hi[1],hi[2],hi[3],hi[4],hij)
            tofile.write(tmpln)
    tofile.close()
    print('complete')
    
def gen_txtfile(pars,fname):
    if type(pars)==tuple or type(pars)==list or type(pars)==np.ndarray:
        pars = {'t0':pars[0],'t1':pars[1],'t2':pars[2],'t3':pars[3],'t4':pars[4],'t5':pars[5],'D':pars[6],'E':pars[7]}
    Hlist = Hbuild(pars)
    Hwrite(Hlist,fname)
            
   
if __name__ == "__main__":
    pars = {'t0':-3.12,'t1':0.355,'t2':-0.010,'t3':0.24,
        't4':0.12,'t5':0.019,'D':-0.008,'E':-0.024} #ELSEVIER
    
    filename = "graphite_test.txt"

    gen_txtfile(pars,filename)
    