#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 13:08:26 2018

@author: ryanday
"""

from matplotlib import rc
from matplotlib import rcParams
import os
rcParams.update({'figure.autolayout':True})

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import tkinter as Tk
from tkinter import messagebox

import ubc_tbarpes.build_lib as build_lib
import ubc_tbarpes.orbital as olib
import ubc_tbarpes.inside as pped
import ubc_tbarpes.atomic_mass as am

class lattice_generator():
    
    def __init__(self):
        self.avecs = np.identity(3)
        self.lpts = region(1)
        self.unit_cell = make_lines(self.avecs)
        self.line = []
        self.dots = []
        self.atoms = []
        self.widget_make()
        
        
    def widget_make(self):
        self.root = Tk.Tk()
        self.root.wm_title('UBC_TBARPES LATTICE BUILDER')
        fig1 = Figure(figsize=(10,4))
        ax1 = fig1.add_subplot(111,projection='3d')

        self.canvas = FigureCanvasTkAgg(fig1,master=self.root)
        ax1.mouse_init()
        
        self.plot_lines(ax1) #update to be plot lattice


        self.canvas.show()
        self.canvas.get_tk_widget().grid(row=0,column=0,columnspan=1,rowspan=4)
        
        
###################### LATTICE VECTORS ########################################
        Axlab = Tk.Label(master=self.root,text='Unit Cell Dimensions').grid(row=0,column=1)
        
        A00box = Tk.Entry(master=self.root)
        A00box.insert('end','{:0.04f}'.format(1.0))
        A00box.grid(row=1,column=1)
        
        A01box = Tk.Entry(master=self.root)
        A01box.insert('end','{:0.04f}'.format(0.0))
        A01box.grid(row=1,column=2)
        
        A02box = Tk.Entry(master=self.root)
        A02box.insert('end','{:0.04f}'.format(0.0))
        A02box.grid(row=1,column=3)
        
        A10box = Tk.Entry(master=self.root)
        A10box.insert('end','{:0.04f}'.format(0.0))
        A10box.grid(row=2,column=1)
        
        A11box = Tk.Entry(master=self.root)
        A11box.insert('end','{:0.04f}'.format(1.0))
        A11box.grid(row=2,column=2)
        
        A12box = Tk.Entry(master=self.root)
        A12box.insert('end','{:0.04f}'.format(0.0))
        A12box.grid(row=2,column=3)
        
        A20box = Tk.Entry(master=self.root)
        A20box.insert('end','{:0.04f}'.format(0.0))
        A20box.grid(row=3,column=1)
        
        A21box = Tk.Entry(master=self.root)
        A21box.insert('end','{:0.04f}'.format(0.0))
        A21box.grid(row=3,column=2)
        
        A22box = Tk.Entry(master=self.root)
        A22box.insert('end','{:0.04f}'.format(1.0))
        A22box.grid(row=3,column=3)
        
        def _avupdate():
            self.avecs[0] = np.array([float(A00box.get()),float(A01box.get()),float(A02box.get())])
            self.avecs[1] = np.array([float(A10box.get()),float(A11box.get()),float(A12box.get())])
            self.avecs[2] = np.array([float(A20box.get()),float(A21box.get()),float(A22box.get())])
            self.unit_cell = make_lines(self.avecs)
            if len(self.atoms)>0:
                for ai in self.atoms:
                    if type(ai.fpos)==np.ndarray:
                        ai.pos = np.dot(ai.fpos,self.avecs)

            self.plot_lines(ax1)
            self.plot_atoms(ax1)
            self.canvas.draw()
        
        
        A_upd = Tk.Button(master=self.root,text='UPDATE LATTICE',command=_avupdate)
        A_upd.grid(row=4,column=3)
        
###################### LATTICE VECTORS ########################################

        def _add_atom():
            add_orbital_win(self)
            self.plot_atoms(ax1)
        
        def _remove_atom():
            print('REMOVING AN ATOM')
        
        add_button = Tk.Button(master=self.root,text='ADD',command=_add_atom)
        add_button.grid(row=4,column=1)
        
        rem_button = Tk.Button(master=self.root,text='REMOVE',command=_remove_atom)
        rem_button.grid(row=4,column=2)
        
    
        
        def _quit():
            self.root.quit()
            self.root.destroy()
        
        
        qu_button = Tk.Button(master=self.root,text='EXIT',command=_quit)
        qu_button.grid(row=5,column=3,columnspan=1)
        
        Tk.mainloop()
        
        
        
        
        
    def plot_lines(self,ax):
        self.line = self.refresh_cell()
        for ii in range(len(self.unit_cell)):
            tmp, = ax.plot(self.unit_cell[ii,:,0],self.unit_cell[ii,:,1],self.unit_cell[ii,:,2],c='k')   
            self.line.append(tmp)
        self.canvas.draw()
        
    def plot_atoms(self,ax):
        self.dots = self.refresh_atoms()
        for ii in range(len(self.atoms)):
            plot_pts = np.array([self.atoms[ii].pos+pi for pi in np.dot(self.lpts,self.avecs)])
            tmp = ax.scatter(plot_pts[:,0],plot_pts[:,1],plot_pts[:,2],c=self.atoms[ii].colour,s=self.atoms[ii].size)

#            tmp = ax.scatter([self.atoms[ii].pos[0]],[self.atoms[ii].pos[1]],[self.atoms[ii].pos[2]],c=self.atoms[ii].colour,s=self.atoms[ii].size)
            self.dots.append(tmp)
        self.canvas.draw()
        
        
    def refresh_cell(self):
        if len(self.line)>0:
            for li in self.line:
                li.remove()
        return []

    
    def refresh_atoms(self):
        if len(self.dots)>0:
            for di in self.dots:
                di.remove()
        return []

def make_lines(avec):
    box = pped.parallelepiped(avec)
    return box.define_lines()

def parse_orbitals(ostr):
    lproj = {'s':0,'p':1,'d':2}
    std = {0:['0'],1:['1x','1y','1z'],2:['2xy','2xz','2yz','2XY','2ZR']}
    orbs = []
    ostr = ostr.split(',')
    for oi in ostr:
        try:
            n = int(oi[0])
        except ValueError:
            print('All orbitals require a principal quantum number n. Entries must be of form nl or nlXX')
            next
        try:
            l=lproj[oi[1]]
        except ValueError:
            try:
                l = int(oi[1])
        
            except ValueError:
                print('All orbitals require an orbital quantum number l. Entries must be of form nl or nlXX')
                next
        
        proj = [oi[2:]]
        if proj[0]=='':
            try:
                proj = std[l]
            except KeyError:
                print('Orbital must be of s, p, or d type (0<=l<=2)')
                next
        
        orbs+=['{:d}{:s}'.format(n,pi) for pi in proj]
    
    return orbs
        
        
            
        
            
        
    



def add_orbital_win(parent):
    cwin = Tk.Toplevel()
    cwin.wm_title('ADD ATOM TO BASIS')       


    Z_lab = Tk.Label(master=cwin,text='Element').grid(row=0,column=0)
        
    Zbox = Tk.Entry(master=cwin)
    Zbox.insert('end','{:d}'.format(1))
    Zbox.grid(row=0,column=0)
    
    P_lab = Tk.Label(master=cwin,text='Position').grid(row=1,column=0)
    
    opt = ['Fractional','Absolute']
    unit_var = Tk.StringVar(cwin)
    unit_var.set(opt[0])
    
    P_units = Tk.OptionMenu(cwin,unit_var,*opt)
    P_units.grid(row=1,column=1)
    
    xbox = Tk.Entry(master=cwin)
    xbox.insert('end','{:0.04f}'.format(0.0))
    xbox.grid(row=2,column=0)
    
    ybox = Tk.Entry(master=cwin)
    ybox.insert('end','{:0.04f}'.format(0.0))
    ybox.grid(row=2,column=1)
    
    zbox = Tk.Entry(master=cwin)
    zbox.insert('end','{:0.04f}'.format(0.0))
    zbox.grid(row=2,column=2)
    
    o_lab = Tk.Label(master=cwin,text='Orbitals:').grid(row=3,column=0)
    
    o_entry = Tk.Entry(master=cwin)
    o_entry.grid(row=4,column=0)
    
    
    
    
    def _add_atom():
        Z = 0
        try:
            Z = int(Zbox.get())
        except ValueError:
            Z = int(am.get_num_from_el(Zbox.get()))
        
        orbs = parse_orbitals(o_entry.get())

        pos = np.array([float(xbox.get()),float(ybox.get()),float(zbox.get())])
        fpos = (False)
        
        if unit_var.get()=='Fractional':
            fpos = (True,pos)
            pos = np.dot(pos,parent.avecs)
        
        
        new_atom = atom(Z,orbs,pos,len(parent.atoms),fpos)
        parent.atoms.append(new_atom)
        parent.atoms[-1]._print_summary()
        
        
        
        
    add_button = Tk.Button(master=cwin,text='ADD',command=_add_atom)
    add_button.grid(row=5,column=0)


    def _quit_sub():
        cwin.quit()
        cwin.destroy()             
            
    close_button = Tk.Button(master= cwin,text='EXIT',command=_quit_sub)
    close_button.grid(row=5,column=1,columnspan=1)
    cwin.mainloop()



class atom:
    
    def __init__(self,Z,orbs,pos,identity,fpos=(False)):
        
        self.Z = Z
        if fpos[0]==True:
            self.fpos = fpos[1]
        self.identity = identity
        self.orbs = orbs
        self.pos = pos
        self.size = np.sqrt(self.Z)*30
        self.colour = hex_col()


    def _print_summary(self):
        
        print('------ATOM {:d} SUMMARY-------'.format(self.identity))
        print('ATOMIC NUMBER: {:d}'.format(self.Z))
        print('POSITION: {:0.03f} {:0.03f} {:0.03f}'.format(self.pos[0],self.pos[1],self.pos[2]))
        print('ORBITALS: ',self.orbs)
        print('---------------------------')
        
        
        

def hex_col():
    rgb = (int(np.random.random()*255) for i in range(3))
    return '#{:02X}{:02X}{:02X}'.format(*rgb)    



def region(num):
    '''
    Generate a symmetric grid of points in number of lattice vectors. The tacit assumption is a 3 dimensional lattice
    args: num -- integer--grid will have size 2*num+1 in each direction
    returns numpy array of size ((2*num+1)**3,3) with centre value of first entry of (-num,-num,-num),...,(0,0,0),...,(num,num,num)
    '''
    num_symm = 2*num+1
    return np.array([[int(i/num_symm**2)-num,int(i/num_symm)%num_symm-num,i%num_symm-num] for i in range((num_symm)**3)])



    


if __name__=="__main__":
    
    win = lattice_generator()
    

    