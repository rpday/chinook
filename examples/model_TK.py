# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 12:44:39 2018

@author: rday



Tkinter GUI for playing with graphite model
________________________________________________________
|0        GRAPHITE MODEL PLOTTER                        |
|1 |---------------------------------|   t0 |         | |
|2 |                                 |   t1 |         | | 
|3 |                                 |   t2 |         | |
|4 |                                 |   t3 |         | |
|5 |               PLOT              |   t4 |         | |
|6 |                                 |   t5 |         | |
|7 |                                 |   D  |         | |
|8 |                                 |   E  |         | |
|9 |---------------------------------|                  |
|10|Kx          Ky         Kz             | UPDATE |    |
|11|        |  |        |  |        |                   |
|12|Kx          Ky         Kz                           |
|13|        |  |        |  |        |     |  QUIT  |    |
|14|NK          Emin       Emax                         |
|15|        |  |        |  |        |                   |
|_______________________________________________________|


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

import graphite_shell as graphite
import ubc_tbarpes.build_lib as blib

class model_interface:
    
    def __init__(self):
        self.root = Tk.Tk()
        self.tb_args = (-3.12,0.355,-0.010,0.24,0.12,0.019,-0.008,-0.024) 
        self.klims = (np.array([0,0,0]),np.array([0,1.702,0.0]))
        self.Nk = 200
        self.Elims = (-1,1)
        Kobj = self.build_K()
        self.ax_k = np.linalg.norm(Kobj.kpts,axis=1)
        self.TB = graphite.build_TB(Kobj)
        self.TB.mat_els = precondition_H(self.TB.mat_els)
        self.TB.solve_H()
        self.plot_make()
        
    def plot_make(self):
        self.root.wm_title('GRAPHITE TIGHT-BINDING MODEL')
        fig1 = Figure(figsize=(9,3))
        ax1 = fig1.add_subplot(111)
        self.plot_TB(ax1)
        ax1.set_xlabel('Momentum ($\AA^{-1}$)')
        ax1.set_ylabel('Energy (eV)')
        plt_win = FigureCanvasTkAgg(fig1,master=self.root)
        plt_win.show()
        plt_win.get_tk_widget().grid(row=0,column=0,columnspan=3,rowspan=9)
        

        #####---------------PARAMETER ENTRY---------------#########
        t0lab = Tk.Label(master=self.root,text='t0').grid(row=0,column=3)
        t0box = Tk.Entry(master=self.root)
        t0box.insert('end','{:0.4f}'.format(self.tb_args[0]))
        t0box.grid(row=0,column=4)
        
        t1lab = Tk.Label(master=self.root,text='t1').grid(row=1,column=3)
        t1box = Tk.Entry(master=self.root)
        t1box.insert('end','{:0.4f}'.format(self.tb_args[1]))
        t1box.grid(row=1,column=4)
        
        t2lab = Tk.Label(master=self.root,text='t2').grid(row=2,column=3)
        t2box = Tk.Entry(master=self.root)
        t2box.insert('end','{:0.4f}'.format(self.tb_args[2]))
        t2box.grid(row=2,column=4)
        
        t3lab = Tk.Label(master=self.root,text='t3').grid(row=3,column=3)
        t3box = Tk.Entry(master=self.root)
        t3box.insert('end','{:0.4f}'.format(self.tb_args[3]))
        t3box.grid(row=3,column=4)
        
        t4lab = Tk.Label(master=self.root,text='t4').grid(row=4,column=3)
        t4box = Tk.Entry(master=self.root)
        t4box.insert('end','{:0.4f}'.format(self.tb_args[4]))
        t4box.grid(row=4,column=4)
        
        t5lab = Tk.Label(master=self.root,text='t5').grid(row=5,column=3)
        t5box = Tk.Entry(master=self.root)
        t5box.insert('end','{:0.4f}'.format(self.tb_args[5]))
        t5box.grid(row=5,column=4)
        
        Dlab = Tk.Label(master=self.root,text='D').grid(row=6,column=3)
        Dbox = Tk.Entry(master=self.root)
        Dbox.insert('end','{:0.4f}'.format(self.tb_args[6]))
        Dbox.grid(row=6,column=4)
        
        Elab = Tk.Label(master=self.root,text='E').grid(row=7,column=3)
        Ebox = Tk.Entry(master=self.root)
        Ebox.insert('end','{:0.4f}'.format(self.tb_args[7]))
        Ebox.grid(row=7,column=4)
        #####---------------PARAMETER ENTRY---------------#########
        
        #####---------------PLOT PARAMETER ENTRY---------------#########
        x0lab = Tk.Label(master=self.root,text='Kx(0)').grid(row=10,column=0)
        x0ent = Tk.Entry(master=self.root)
        x0ent.insert('end','0.0')
        x0ent.grid(row=11,column=0)
        
        y0lab = Tk.Label(master=self.root,text='Ky(0)').grid(row=10,column=1)
        y0ent = Tk.Entry(master=self.root)
        y0ent.insert('end','0.0')
        y0ent.grid(row=11,column=1)
        
        z0lab = Tk.Label(master=self.root,text='Kz(0)').grid(row=10,column=2)
        z0ent = Tk.Entry(master=self.root)
        z0ent.insert('end','0.0')
        z0ent.grid(row=11,column=2)
        
        xflab = Tk.Label(master=self.root,text='Kx(-1)').grid(row=12,column=0)
        xfent = Tk.Entry(master=self.root)
        xfent.insert('end','0.0')
        xfent.grid(row=13,column=0)
        
        yflab = Tk.Label(master=self.root,text='Ky(-1)').grid(row=12,column=1)
        yfent = Tk.Entry(master=self.root)
        yfent.insert('end','1.702')
        yfent.grid(row=13,column=1)
        
        zflab = Tk.Label(master=self.root,text='Kz(-1)').grid(row=12,column=2)
        zfent = Tk.Entry(master=self.root)
        zfent.insert('end','0.0')
        zfent.grid(row=13,column=2)
        
        Nlab = Tk.Label(master=self.root,text='Number of K Points').grid(row=14,column=0)
        Nent = Tk.Entry(master=self.root)
        Nent.insert('end','200')
        Nent.grid(row=15,column=0)
        
        Eolab = Tk.Label(master=self.root,text='Minimum E Plot (eV)').grid(row=14,column=1)
        Eoent = Tk.Entry(master=self.root)
        Eoent.insert('end','-1.0')
        Eoent.grid(row=15,column=1)
        
        Eflab = Tk.Label(master=self.root,text='Maximum E Plot (eV)').grid(row=14,column=2)
        Efent = Tk.Entry(master=self.root)
        Efent.insert('end','1.0')
        Efent.grid(row=15,column=2)
        
        
        

        def _update():
            k0 = np.array([float(x0ent.get()),float(y0ent.get()),float(z0ent.get())])
            kf = np.array([float(xfent.get()),float(yfent.get()),float(zfent.get())])
            self.klims = (k0,kf)
            self.Nk = int(Nent.get())
            self.Elims = (float(Eoent.get()),float(Efent.get()))
            self.TB.Kobj = self.build_K()
            self.ax_k = np.linalg.norm(self.TB.Kobj.kpts,axis=1)

            
            self.tb_args = (float(t0box.get()),float(t1box.get()),float(t2box.get()),float(t3box.get()),
                            float(t4box.get()),float(t5box.get()),float(Dbox.get()),float(Ebox.get()))
            self.TB.mat_els = rebuild_H(self.tb_args,self.TB.mat_els)
            self.TB.solve_H()
            self.redraw_plots(fig1,ax1)
            
            print('updating model')
        
        update_button = Tk.Button(master=self.root,text='UPDATE',command=_update)
        update_button.grid(row=10,column=3,columnspan=2)
        
        def _reset():
            self.tb_args = (-3.12,0.355,-0.010,0.24,0.12,0.019,-0.008,-0.024)
            t0box.delete(0,'end')
            t0box.insert('end',self.tb_args[0])
            
            t1box.delete(0,'end')
            t1box.insert('end',self.tb_args[1])
            
            t2box.delete(0,'end')
            t2box.insert('end',self.tb_args[2])
            
            t3box.delete(0,'end')
            t3box.insert('end',self.tb_args[3])
        
            t4box.delete(0,'end')
            t4box.insert('end',self.tb_args[4])
            
            t5box.delete(0,'end')
            t5box.insert('end',self.tb_args[5])
            
            Dbox.delete(0,'end')
            Dbox.insert('end',self.tb_args[6])
            
            Ebox.delete(0,'end')
            Ebox.insert('end',self.tb_args[7])
            
            self.TB.mat_els= rebuild_H(self.tb_args,self.TB.mat_els)
            self.TB.solve_H()
            self.redraw_plots(fig1,ax1)
        
        
        reset_button = Tk.Button(master=self.root,text='RESET MODEL',command=_reset)
        reset_button.grid(row=11,column=3,columnspan=2)
        
        def _autoscale():
            Emin,Emax=self.TB.Eband.min()-0.1,self.TB.Eband.max()+0.1
            self.Elims = (Emin,Emax)
            Eoent.delete(0,'end')
            Eoent.insert('end',self.Elims[0])
            Efent.delete(0,'end')
            Efent.insert('end',self.Elims[-1])
            ax1.set_ylim(Emin,Emax)
            fig1.canvas.draw()
        
        auto_button = Tk.Button(master= self.root,text='AUTOSCALE',command=_autoscale)
        auto_button.grid(row=12,column=3,columnspan=2)
            
        
        def _quit_win():
            self.root.quit()
            self.root.destroy()
            
        quit_button = Tk.Button(master=self.root,text='QUIT',command=_quit_win)
        quit_button.grid(row=13,column=3,columnspan=2)
        
        
        
        def onclick(event):
            ix,iy = event.xdata,event.ydata
            kind = np.where(abs(self.ax_k-ix)==abs(self.ax_k-ix).min())[0][0]
            ki = self.TB.Kobj.kpts[kind]
            print('Coordinates: [{:0.4f}, {:0.4f}, {:0.4f}] 1/A, {:0.4f} eV'.format(ki[0],ki[1],ki[2],iy))
        
        cid = fig1.canvas.mpl_connect('button_press_event',onclick)
        
        Tk.mainloop()
        
        
        
        
    def plot_TB(self,axis):
        self.line = []
        for ii in range(np.shape(self.TB.Eband)[-1]):
            tmp, = axis.plot(self.ax_k,self.TB.Eband[:,ii],c='r')
            self.line.append(tmp)
        axis.set_xlim(self.ax_k[0],self.ax_k[-1])
        axis.set_ylim(*self.Elims)
        
            
    def redraw_plots(self,fig,axis):
        for ii in range(np.shape(self.TB.Eband)[-1]):
            self.line[ii].set_xdata(self.ax_k)
            self.line[ii].set_ydata(self.TB.Eband[:,ii])
        axis.set_xlim(self.ax_k[0],self.ax_k[-1])
        axis.set_ylim(*self.Elims)
        fig.canvas.draw()
        
        
    def build_K(self):
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
    			'pts':[*self.klims],
    			'grain':self.Nk,
    			'labels':[]}
        Kobj = blib.gen_K(Kd)
        return Kobj
    
    

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

    for i in range(len(mats)):
        tmp = mats[i].H
        tmp[:,-1] = np.array(args)[tdic[i]]
        mats[i].H = tmp
    return mats
    
if __name__ == "__main__":
    
    now = model_interface()



