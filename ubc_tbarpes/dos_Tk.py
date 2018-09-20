#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 11:58:18 2018

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

import ubc_tbarpes.dos as dos
import ubc_tbarpes.atomic_mass as am


class chkmk:
    
    def __init__(self,item,master,ostr,indx):
        self.item = item
        self.var = Tk.IntVar()
        self.txt = ostr
        self.chk = Tk.Checkbutton(master=master,text=self.txt,variable=self.var)
        self.chk.var = self.var
        self.indx = indx
    
    def _reset_master(self,master):
        self.chk = Tk.Checkbutton(master=master,text=self.txt,variable=self.var)
        self.chk.var = self.var       
        

        
class dos_interface:
    
    def __init__(self,TB,sigma=0.0):
        self.dos = dos.dos_env(TB,sigma)
        self.dos.TB.Kobj.kpts = self.dos.build_bz(20)
        self.dos.Evals,self.dos.Evecs,self.dos.Nbins =self.dos.prep_bands()
        self.dos.hi = self.dos.solve_dos()
        ###PDOS PREP#####
        self.o_labels = gen_labels(TB.basis)
        self.chks = []
        
        self.plot_make()
        
    def plot_make(self):
        self.root = Tk.Tk()
        self.root.wm_title('UBC_TBARPES DENSITY OF STATES')
        fig1 = Figure(figsize=(10,4))
        ax1 = fig1.add_subplot(111)
        self.plot_dos(ax1)
        self.legend=['Total']
        ax1.legend(self.legend)

        ax1.set_xlabel('Energy (eV)')
        ax1.set_ylabel('Density of States (a.u.)')
        plt_win = FigureCanvasTkAgg(fig1,master=self.root)

        plt_win.show()
        plt_win.get_tk_widget().grid(row=0,column=0,columnspan=10,rowspan=4)
        
        Tk.Label(master=self.root,text='Nk x: ').grid(row=4,column=0)
        Nxbox = Tk.Entry(master=self.root)
        Nxbox.insert('end','{:d}'.format(20))
        Nxbox.grid(row=4,column=1)
        
        Tk.Label(master=self.root,text='Nk y: ').grid(row=4,column=2)
        Nybox = Tk.Entry(master=self.root)
        Nybox.insert('end','{:d}'.format(20))
        Nybox.grid(row=4,column=3)
        
        Tk.Label(master=self.root,text='Nk z: ').grid(row=4,column=4)
        Nzbox = Tk.Entry(master=self.root)
        Nzbox.insert('end','{:d}'.format(20))
        Nzbox.grid(row=4,column=5)
        
        
        def _pdos():
            cwin = Tk.Toplevel()
            cwin.wm_title('PARTIAL DENSITY OF STATES')       
            
            if len(self.chks)==0:
                for ostr in self.o_labels:
                    self.chks.append(chkmk(len(self.chks),cwin,ostr,self.o_labels[ostr]))
            if len(self.chks)>0:
                for ii in range(len(self.chks)):
                    self.chks[ii]._reset_master(cwin)
                    self.chks[ii].chk.grid(row=np.mod(self.chks[ii].item,10),column=int(self.chks[ii].item/10))
            
            def _update():
                self.dos.pdos = []
                self.legend = self.legend[:1]
                for ci in range(len(self.chks)):
                    tmp_bool = self.chks[ci].chk.var.get()
                    if tmp_bool:
                        self.dos.pdos.append(self.dos.calc_pdos(self.chks[ci].indx))
                        self.legend.append(self.chks[ci].txt)
                self.plot_pdos(fig1,ax1)
                cwin.quit()
                cwin.destroy()

            
            up_button = Tk.Button(master=cwin,text='COMPUTE PDOS',command=_update)
            up_button.grid(row=10,column=0,columnspan=1)
            
            def _rem_pdos():
                self.dos.pdos = []
                self.legend = self.legend[:1]
                ax1.cla()
                self.plot_dos(ax1)
                ax1.legend(self.legend)
                fig1.canvas.draw()

                
                ### NEED TO CLEAR PLOT HERE, LEAVING ONLY THE DOS BEHIND
                cwin.quit()
                cwin.destroy()
            
            def _quit_sub():
                cwin.quit()
                cwin.destroy()
                
            rem_button = Tk.Button(master=cwin,text='REMOVE PDOS',command=_rem_pdos)
            rem_button.grid(row=10,column=1,columnspan=1)
            
            close_button = Tk.Button(master= cwin,text='EXIT',command=_quit_sub)
            close_button.grid(row=10,column=2,columnspan=1)
            cwin.mainloop()
#        
        
        def _remesh():
            try:
                Nx,Ny,Nz = int(Nxbox.get()),int(Nybox.get()),int(Nzbox.get())
                if Nx>0 and Ny>0 and Nz>0:
                    print(Nx,Ny,Nz)
                    self.dos.do_dos((Nx,Ny,Nz))
                else:
                    #User can also pass a single N value by setting both other numbers to negative values
                    N = int(np.array([Nx,Ny,Nz]).max())
                    self.dos.do_dos(N)

            except TypeError:
                print('Numeric Entries only for K-mesh scaling!')
                return False
            #modify next line to include the projections, once defined
            ax1.cla()
            self.legend = self.legend[:1]
            self.plot_dos(ax1)
            ax1.legend(self.legend)
            fig1.canvas.draw()
            
        
        
        def _quit():
            self.root.quit()
            self.root.destroy()
            
        def _save():
            fnm = Tk.filedialog.asksaveasfile(mode='w',defaultextension='.png')
            fig1.savefig(fnm.name,transparent=True,dpi=300)
            fnm.close()
            
        def _infobox():
            Tk.messagebox.showinfo("About","This panel provides a convenient interface for calculating density (and partial density) of states. The Brillouin Zone can be divided evenly along all three directions by specifying a single number and setting the other two to -1. The PDOS desired can be selected in the PDOS pop-up, where all possible orbital projections are given. The plot can also be saved to file, following the SAVE button.")
        
        pd_button = Tk.Button(master=self.root,text='PDOS',command=_pdos)
        pd_button.grid(row=5,column=0,columnspan=1)
        
        re_button = Tk.Button(master=self.root,text='UPDATE K-MESH',command=_remesh)
        re_button.grid(row=5,column=1,columnspan=1)
        
        sv_button = Tk.Button(master=self.root,text='SAVE',command=_save)
        sv_button.grid(row=5,column=2,columnspan=1)
        
        inf_button = Tk.Button(master=self.root,text='INFO',command=_infobox)
        inf_button.grid(row=5,column=3,columnspan=1)
        
        qu_button = Tk.Button(master=self.root,text='EXIT',command=_quit)
        qu_button.grid(row=5,column=4,columnspan=1)
        
        
        Tk.mainloop()
        
        
    def plot_dos(self,ax):        
        self.line = []
        tmp, = ax.plot(self.dos.hi[1][:-1],self.dos.hi[0])
        ax.set_xlim(self.dos.hi[1][0],self.dos.hi[1][-2])
        self.line.append(tmp)

    def plot_pdos(self,fig,ax):
        ax.cla()
        self.plot_dos(ax)
        for ii in range(len(self.dos.pdos)):
            tmp, = ax.plot(self.dos.hi[1][:-1],self.dos.pdos[ii][0])
            self.line.append(tmp)
        ax.legend(self.legend)
        fig.canvas.draw()
        



def gen_labels(basis):
    od = {'0':'s','1':'p','2':'d','3':'f'}
    
    label_dict = {}
    for o in basis:
        element = am.get_el_from_number(o.Z)
        l = od[o.label[1]]
        n = int(o.label[0])
        ostr = '{:s}{:d} {:d}{:s}{:s}'.format(element,o.atom,n,l,o.label[2:])
        if ostr not in label_dict:
            label_dict[ostr] = [o.index]
        else:
            label_dict[ostr].append(o.index)
    return label_dict


        
        

        
        
        
        
