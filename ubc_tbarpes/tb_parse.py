# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 09:40:00 2018

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
Utility script for loading tight-binding model from wannier90 (seedname_tb.dat)


'''
import numpy as np


class system:
    
    def __init__(self,avec,pos,infile,outfile):
        self.avec=avec
        self.pos = pos
        self.filename = infile
        self.Hlist = self.load_file()
        self.out = outfile
        
    def load_file(self):
        cvec = np.zeros(3)
        Hlist = []
        with open(self.filename,'r') as infile:
            for line in infile:
                line_list = line.split()
                if len(line_list)==3:                    
                    cvec = np.array([float(line_list[0]),float(line_list[1]),float(line_list[2])])
                elif len(line_list)==4:
                    i,j=int(line_list[0])-1,int(line_list[1])-1
                    Hij = float(line_list[2])+1.0j*float(line_list[3])
                    Rij = self.pos[i]-self.pos[j]-cvec
                    Hlist.append([i,j,Rij[0],Rij[1],Rij[2],Hij])
        return Hlist
    
    def write_H(self):
        
        with open(self.out,'w') as outfile:
            for h in self.Hlist:
                pm = '+-'[np.imag(h[5])<0]
                tmpline = '{:d},{:d},{:0.4f},{:0.4f},{:0.4f},{:0.4f}{:s}{:0.4f}j\n'.format(h[0],h[1],h[2],h[3],h[4],np.real(h[5]),pm,abs(np.imag(h[5])))
                outfile.write(tmpline)
        outfile.close()
        print('Done generating Hamiltonian file. Matrix elements written to {:s}'.format(self.out))
        
        

if __name__=="__main__":
     
    bohr = 0.529177249 #bohr to Angstrom
    a,c=7.12011*bohr,11.9338*bohr
    avecs = np.array([[a,0,0],[0,a,0,],[0,0,c]])
    
    pos = np.array([avecs[0]*0.75+0.25*avecs[1],0.25*avecs[0]+0.75*avecs[1],0.25*avecs[0]+0.25*avecs[1]+0.2326*avecs[2],0.75*avecs[0]+0.75*avecs[1]+0.76374*avecs[2]]) # LiFeAs
    l_list = np.array([2,2,1,1])
    full_pos = np.array([pos[x] for x in range(4) for m in range(2*l_list[x]+1)])
    
    
    infile = 'lifeas_tb.dat'
    outfile = 'lifeas_w90tb.txt'
    
    model = system(avecs,full_pos,infile,outfile)
    model.write_H()

    