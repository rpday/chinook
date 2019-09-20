#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Created on Thu Feb 22 11:34:32 2018

#@author: rday

#MIT License

#Copyright (c) 2018 Ryan Patrick Day

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

import matplotlib
from matplotlib import rc
from matplotlib import rcParams
import os
rcParams.update({'figure.autolayout':True})


import numpy as np

from matplotlib.figure import Figure
import matplotlib.cm as cm

try:
    import tkinter as Tk
    from tkinter import messagebox
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    tk_found = True
except ModuleNotFoundError:
    print('tkinter not found, please load for interactive map plotting')
    tk_found = False



import chinook.intensity_map as imap


    
    
def tk_query():
    if tk_found:
        return True
    else:
        return False

    
if tk_found:

    class plot_intensity_interface:
        '''
        Tkinter-based plotting interface. Interactive, allows for generation
        and exploration of intensity maps, given an existing TB and experiment
        object.
        '''
        
        def __init__(self,experiment):
            self.root = Tk.Tk()
            self.experiment = experiment
            print('Initializing spectral function...')
            try:
                _,self.Imat = self.experiment.spectral()
            except AttributeError:
                print('Matrix elements have not yet been computed. Initializing matrix elements now...')
                self.experiment.datacube()
                print('Matrix element calculation complete.')
                _,self.Imat = self.experiment.spectral()
                
            map_pars = (0,self.Imat,self.experiment.cube,self.experiment.kz,self.experiment.T,self.experiment.hv,self.experiment.pol,self.experiment.dE,self.experiment.dk,self.experiment.SE_args,self.experiment.sarpes,self.experiment.ang)
    
            self.Imat_dict = {'I_0':imap.intensity_map(*map_pars)}
            self.x = self.experiment.cube[0]
            self.y = self.experiment.cube[1]
            self.w = self.experiment.cube[2]
            self.dx,self.dy,self.dw = (self.x[1]-self.x[0])/self.x[2],(self.y[1]-self.y[0])/self.y[2],(self.w[1]-self.w[0])/self.w[2]
            print('Initializing interface...')
            self.plot_make()
    
        def plot_make(self):
               
            self.root.wm_title('CHINOOK DATA PLOTTER')
    
            fig1 = Figure(figsize=(5,5))
            fig2 = Figure(figsize=(5,5))
            fig3 = Figure(figsize=(5,5))
            
            rc('font',**{'family':'serif','serif':['Palatino'],'size':12})
            rc('text',usetex = False) 
    
            ax1= fig1.add_subplot(111)
            ax2= fig2.add_subplot(111)
            ax3= fig3.add_subplot(111)
            sp1 = ax1.imshow(self.Imat[:,:,int(self.w[2]/2)],extent=[self.x[0],self.x[1],self.y[1],self.y[0]],cmap=cm.magma) ##IMSHOW HAS INVERTED ORIGIN CONVENTION TO PCOLORMESH--ORIGIN IS AT TOP-LEFT INSTEAD OF BOTTOM-LEFT
            sp2 = ax2.imshow(self.Imat[:,int(self.x[2]/2),:],extent=[self.w[0],self.w[1],self.y[1],self.y[0]],cmap=cm.magma)
            sp3 = ax3.imshow(self.Imat[int(self.y[2]/2),:,:],extent=[self.w[0],self.w[1],self.x[1],self.x[0]],cmap=cm.magma)
            sp1.set_clim(vmin=0,vmax=self.Imat.max()*1.05)
            sp2.set_clim(vmin=0,vmax=self.Imat.max()*1.05)
            sp3.set_clim(vmin=0,vmax=self.Imat.max()*1.05)
            try:
                asp1 = (self.x[1]-self.x[0])/(self.y[1]-self.y[0])
            except ZeroDivisionError:
                asp1=1.0
            try:
                asp2 = (self.w[1]-self.w[0])/(self.y[1]-self.y[0])
            except ZeroDivisionError:
                asp2=1.0
            try:
                asp3 = (self.w[1]-self.w[0])/(self.x[1]-self.x[0])
            except ZeroDivisionError:
                asp3 =1.0
            ax1.set_aspect(asp1)
            ax2.set_aspect(asp2)
            ax3.set_aspect(asp3)
            
            ax1.tick_params(axis='both',labelsize=12)
            ax2.tick_params(axis='both',labelsize=12)
            ax3.tick_params(axis='both',labelsize=12)
            
            ax1.set_ylabel('Momentum y ($\AA^{-1}$)')
            ax1.set_xlabel('Momentum x ($\AA^{-1}$)')
            ax2.set_ylabel('Momentum y ($\AA^{-1}$)')
            ax2.set_xlabel('Energy (eV)')
            ax3.set_ylabel('Momentum x ($\AA^{-1}$)')
            ax3.set_xlabel('Energy (eV)')
    
            plt_win_1 = FigureCanvasTkAgg(fig1,master=self.root)
            plt_win_1.show()
            plt_win_1.get_tk_widget().grid(row=0,column=0,columnspan=3,rowspan=3)
            
            plt_win_2 = FigureCanvasTkAgg(fig2,master=self.root)
            plt_win_2.show()
            plt_win_2.get_tk_widget().grid(row=0,column=3,columnspan=3,rowspan=3)
    
            plt_win_3 = FigureCanvasTkAgg(fig3,master=self.root)
            plt_win_3.show()
            plt_win_3.get_tk_widget().grid(row=0,column=6,columnspan=3,rowspan=3)
    
    
            def _quit_win():
                self.root.quit()
                self.root.destroy()
        
            def _update_slice(event):
                sv1 = int((float(slide_1.get())-self.w[0])/(self.dw))
                sv2 = int((float(slide_2.get())-self.x[0])/(self.dx))
                sv3 = int((float(slide_3.get())-self.y[0])/(self.dy))
        
                sp1.set_data(self.Imat[:,:,sv1])
                sp2.set_data(self.Imat[:,sv2,:])
                sp3.set_data(self.Imat[sv3,:,:])
                fig1.canvas.draw()
                fig2.canvas.draw()
                fig3.canvas.draw()
        
            s1_label = Tk.Label(master=self.root,text="Energy (eV)").grid(row=3,column=0)
            slide_1 = Tk.Scale(master=self.root,from_=self.w[0],to=(self.w[1]-self.dw),orient='horizontal',resolution=self.dw,command=_update_slice)
            slide_1.grid(row=3,column=1,sticky='W')
    
            s2_label = Tk.Label(master=self.root,text="Momentum Kx (1/Å)").grid(row=3,column=3)
            slide_2 = Tk.Scale(master=self.root,from_=self.x[0],to=(self.x[1]-self.dx),orient='horizontal',resolution=self.dx,command=_update_slice)
            slide_2.grid(row=3,column=4,sticky='W')
    
            s3_label = Tk.Label(master=self.root,text="Momentum K_y (1/Å)").grid(row=3,column=6)
            slide_3 = Tk.Scale(master=self.root,from_=self.y[0],to=(self.y[1]-self.dy),orient='horizontal',resolution=self.dy,command=_update_slice)
            slide_3.grid(row=3,column=7,sticky='W')
            
            
            def _customize():
                
                cwin = Tk.Toplevel()
                cwin.wm_title('CUSTOMIZE MAP DISPLAY')
                #List of available Maps
                list_lab = Tk.Label(master=cwin,text='Available maps: ').grid(row=0,column=0)
                mat_list = (d for d in self.Imat_dict)
                mat_listbox = Tk.Listbox(master=cwin)
                mat_listbox.grid(row=1,column=0,columnspan=4)
                for d in list(enumerate(self.Imat_dict)):
                    mat_listbox.insert("end",d[1])
                #Spin Vector
                spin_label=Tk.Label(master=cwin,text='Spin Axis: ').grid(row=2,column=0)
                sx = Tk.Entry(master=cwin)
                sy = Tk.Entry(master=cwin)
                sz = Tk.Entry(master=cwin)
                
                sx.insert('end','0.0')
                sy.insert('end','0.0')
                sz.insert('end','1.0')
                
                sx.grid(row=2,column=1)
                sy.grid(row=2,column=2)
                sz.grid(row=2,column=3)
                
                #Spin projection -- -1, None, 1
                proj_label = Tk.Label(master=cwin,text='Spin Projection: ').grid(row=3,column=0)
                proj_list = ("Down","Up","None")
                proj_var = Tk.StringVar()
                proj_var.set(proj_list[2])
                proj_opt = Tk.OptionMenu(cwin,proj_var,*proj_list)
                proj_opt.grid(row=3,column=1)
                #Polarization Vector
                pol_label = Tk.Label(master=cwin,text='Polarization: ').grid(row=4,column=0)
                pol_x = Tk.Entry(master=cwin)
                pol_y = Tk.Entry(master=cwin)
                pol_z = Tk.Entry(master=cwin)
                
                pol_x.insert('end','1.0+0.0j')
                pol_y.insert('end','0.0+0.0j')
                pol_z.insert('end','0.0+0.0j')
                
                pol_x.grid(row=4,column=1)
                pol_y.grid(row=4,column=2)
                pol_z.grid(row=4,column=3)
                #Name of New Map -- option of string or Blank
                
                ###SELF ENERGY
                def _self_energy():
                    
                    sewin = Tk.Toplevel()
                    sewin.wm_title('SELF ENERGY')
                    
                    SE_label = Tk.Label(master=sewin,text='Im(Self-Energy)').grid(row=0,column=0)
                    SE_entry = Tk.Entry(master=sewin)
                    SE_entry.grid(row=0,column=1)
    
                    def _se_infobox():
                        messagebox.showinfo('Information','The imaginary part of the self energy can be entered here. The absolute value of the function will be taken, preserving particle-hole symmetry and causality.',parent=sewin)
                
                    def _quit_sub():
                        sewin.quit()
                        sewin.destroy()
                    
                    se_info_button = Tk.Button(master=sewin,text='Info',command=_se_infobox)
                    se_info_button.grid(row=1,column=0,columnspan=1)        
                    
                    SEquit = Tk.Button(master=sewin,text='Quit',command=_quit_sub)
                    SEquit.grid(row=1,column=1)
                    
                    sewin.mainloop()
                
                SE_button = Tk.Button(master=cwin,text='Self-Energy',command=_self_energy)
                SE_button.grid(row=5,column=0)
                
                mat_nm_label = Tk.Label(master=cwin,text='Map Name: ').grid(row=6,column=0)
                mat_nm = Tk.Entry(master=cwin)
                mat_nm.grid(row=6,column=1)
            
                
                #Add a new map to the list of available maps
                def _add_map():
                    
                    tmp_dict = {}
                    #Define the polarization vector
                    tmp_dict['pol'] = xyz(pol_x.get(),pol_y.get(),pol_z.get())
                    
                    proj_choice = proj_var.get()
                    if proj_choice=="None":
                        tmp_dict['spin']=None
                    else:
                        
                        tmp_dict['spin'] = [-1 if proj_choice=="Down" else 1,np.array([float(sx.get()),float(sy.get()),float(sz.get())])]
                    mat_name = mat_nm.get() if mat_nm.get()!="" else "I_{:d}".format(len(self.Imat_dict))
    
                    map_pars = (self.experiment.cube,self.experiment.kz,self.experiment.T,self.experiment.hv,tmp_dict['pol'],self.experiment.dE,self.experiment.dk,self.experiment.SE_args,tmp_dict['spin'],self.experiment.ang)
                    _,Imap = self.experiment.spectral(ARPES_dict = tmp_dict)
                    
                    self.Imat_dict[mat_name] = imap.intensity_map(len(self.Imat_dict.keys()),Imap,*map_pars)
    #                self.meta[mat_name] = tmp_dict #Save the parameters associated with a given calculation for use in export
                    
                    mat_listbox.insert("end",mat_name)            
                
                add_button = Tk.Button(master=cwin,text ='Add Map ',command=_add_map)
                add_button.grid(row=6,column=2,columnspan=1)            
                
                #add option of operating on datasets
                def _gen_map():
                    st_raw = op_entry.get()
                    placeholder  = ''
                    for d in self.Imat_dict:
                        before = st_raw
                        replacement = 'self.Imat_dict["{:s}"].Imat'.format(d)
                        st_raw = st_raw.replace(d,replacement)
                        if st_raw!=before:
                            placeholder = d
                        self.Imat_dict[d].Imat+=  abs(self.Imat_dict[d].Imat[np.nonzero(self.Imat_dict[d].Imat)]).min()*10**-4#avoid divergence for division if zeros present
                    st_raw = st_raw.replace("SQRT","np.sqrt")
                    st_raw = st_raw.replace("COS","np.cos")
                    st_raw = st_raw.replace("SIN","np.sin")
                    st_raw = st_raw.replace("TAN","np.tan")
                    st_raw = st_raw.replace("EXP","np.exp")
                    st_raw = st_raw.replace("LOG","np.log")
                    st_raw = st_raw.replace("^","**")
                    
                    tmp_mat = eval(st_raw)
                    map_nm = mat_nm.get() if (mat_nm.get()!="" or bool(sum([mat_nm==d for d in self.Imat_dict]))) else "I_{:d}".format(len(self.Imat_dict))
                    mat_listbox.insert("end",map_nm)
    #                tmp_mat = eval(replacement).copy()
                    self.Imat_dict[map_nm] = self.Imat_dict[placeholder].copy()
                    self.Imat_dict[map_nm].Imat = tmp_mat.copy()
                    self.Imat_dict[map_nm].notes = 'Intensity calculated as: {:s}'.format(st_raw)
    
                    
            
                op_label = Tk.Label(master=cwin,text="Operate: ").grid(row=7,column=0)
                op_entry = Tk.Entry(master=cwin)
                op_entry.grid(row=7,column=1)
                op_gen = Tk.Button(master=cwin,text = "Generate",command=_gen_map)
                op_gen.grid(row=7,column=2)
                
                def _plot_map():
                    selected = mat_listbox.curselection()[0]
                    try:
                        
                        self.Imat = self.Imat_dict[mat_listbox.get(selected)].Imat
                        sv1 = int((float(slide_1.get())-self.w[0])/(self.dw))
                        sv2 = int((float(slide_2.get())-self.x[0])/(self.dx))
                        sv3 = int((float(slide_3.get())-self.y[0])/(self.dy))
        
                        sp1.set_data(self.Imat[:,:,sv1])
                        sp2.set_data(self.Imat[:,sv2,:])
                        sp3.set_data(self.Imat[sv3,:,:])
                        sp1.set_clim(vmin=self.Imat.min(),vmax=self.Imat.max())
                        sp2.set_clim(vmin=self.Imat.min(),vmax=self.Imat.max())
                        sp3.set_clim(vmin=self.Imat.min(),vmax=self.Imat.max())
                        fig1.canvas.draw()
                        fig2.canvas.draw()
                        fig3.canvas.draw()
                        cwin.quit()
                        cwin.destroy()
                        
                    except IndexError:
                        print('Please select a map to plot before attempting to plot.')
                
                plot_button = Tk.Button(master=cwin,text="Plot Selection", command=_plot_map)
                plot_button.grid(row=8,column=0)
                
                def _infobox():
                    messagebox.showinfo("Information","This panel allows to generate intensity maps with or without\n spin projection, for arbitrary polarization. You can also enter a\n function over several intensity maps, referencing them by names indicated in the list of available maps. Standard polynomial\n operations as well as SQRT, COS, SIN, TAN, EXP, and LOG functions\n can be passed.",parent=cwin) 
                
                info_button = Tk.Button(master=cwin,text='Info',command=_infobox)
                info_button.grid(row=8,column=2)
                
                def _quit_sub():
                    cwin.quit()
                    cwin.destroy()
                
                close_button = Tk.Button(master= cwin,text='Close Window',command=_quit_sub)
                close_button.grid(row=8,column=3,columnspan=1)
                cwin.mainloop()
                
                
            
            custmap_button = Tk.Button(master=self.root,text = 'Custom Map', command = _customize)
            custmap_button.grid(row=5,column=0)
            
            clabel = Tk.Label(master=self.root,text='Colourmap: ').grid(row=4,column=0)
            cmaps = ('Spectral','Blue-White-Red','Magma','Inferno','Black-to-White','White-to-Black')
            cmdict = {'Spectral':cm.Spectral,'Blue-White-Red':cm.bwr,'Magma':cm.magma,'Inferno':cm.inferno,'Black-to-White':cm.Greys_r,'White-to-Black':cm.Greys}
            cmap_var = Tk.StringVar()
            cmap_var.set(cmaps[2])
            cmap_opt = Tk.OptionMenu(self.root,cmap_var,*cmaps)
            cmap_opt.grid(row=4,column=1)
                
            vmin_label = Tk.Label(master=self.root,text='Scale Min: ').grid(row=4,column=2)
            vmin_ent = Tk.Entry(master=self.root)
            vmin_ent.grid(row=4,column=3)
            vmin_ent.insert('end','Auto')
            vmax_label = Tk.Label(master=self.root,text='Scale Max: ').grid(row=4,column=4)
            vmax_ent = Tk.Entry(master=self.root)
            vmax_ent.grid(row=4,column=5)
            vmax_ent.insert('end','Auto')
                
    
            def _update():
                doit = True
                try:
                    vmin = float(vmin_ent.get())
                except ValueError:
                    if vmin_ent.get().upper()=='AUTO':
                        vmin = self.Imat.min()
                    else:
                        print( 'Please enter a numeric value for Scale Minimum ')
                        doit = False
                try:
                    vmax = float(vmax_ent.get())
                except ValueError:
                    if vmax_ent.get().upper()=='AUTO':
                        vmax = self.Imat.max()
                    else:
                        print ('Please enter a numeric value for Scale Maximum ')
                        doit = False
                if vmin>=vmax:
                    doit= False
                if doit:
                    sp1.set_clim(vmin=vmin,vmax=vmax)
                    sp2.set_clim(vmin=vmin,vmax=vmax)
                    sp3.set_clim(vmin=vmin,vmax=vmax)
                    sp1.set_cmap(cmdict[cmap_var.get()])
                    sp2.set_cmap(cmdict[cmap_var.get()])
                    sp3.set_cmap(cmdict[cmap_var.get()])
                    fig1.canvas.draw()
                    fig2.canvas.draw()
                    fig3.canvas.draw()
                        
            col_up = Tk.Button(master=self.root,text='Update Colourscale',command=_update)
            col_up.grid(row=4,column=6)
            
            def _infobox_main():
                messagebox.showinfo("Information","This plotting GUI allows for the user to scan through\n different slices of the momentum-energy cube generated\n in an ARPES experiment. The 'Custom Map' panel allows you to\n change the polarization and spin projection (for Spin-ARPES) as well as \n to produce composite maps generated as a function of maps with different experimental parameters.",parent=self.root) 
                
            main_info_button = Tk.Button(master=self.root,text='Info',command=_infobox_main)
            main_info_button.grid(row=5,column=1)
            
            def _exp_win():
                
                ewin = Tk.Toplevel()
                ewin.wm_title('EXPORT MAPS TO FILE')
    #            List of available Maps
                list_lab = Tk.Label(master=ewin,text='Available maps: ').grid(row=0,column=0)
                mat_list = (d for d in self.Imat_dict)
                mat_listbox = Tk.Listbox(master=ewin)
                mat_listbox.grid(row=1,column=0,columnspan=4)
                for d in list(enumerate(self.Imat_dict)):
                    mat_listbox.insert("end",d[1])
                
                def _sel_dir():
                    
                     self.destination = Tk.filedialog.askdirectory()
                     
                dir_select =Tk.Button(master=ewin,text='Select Directory',command =_sel_dir)
                dir_select.grid(row=2,column=0)
                name_label = Tk.Label(master=ewin,text='File Lead: ').grid(row=3,column=0)
                name_entry = Tk.Entry(master=ewin,text='UNTITLED')
                name_entry.grid(row=3,column=1)
                
                def _export_now():
                    
                    if len(name_entry.get())>0:
                        file_lead=name_entry.get()
                    else:
                        file_lead = 'UNTITLED'
                    filename = self.destination +'/'+ file_lead
                    
                    metafile = filename + '_params.txt'
                    
                    ind_map = mat_listbox.curselection()[0]
                    map_choice = mat_listbox.get(ind_map)
    #                self.Imat_dict[map_choice].write_meta(metafile)
                    self.Imat_dict[map_choice].save_map(filename)
    #                self.expmnt.write_params(self.meta[map_choice],parfile)
    #                self.expmnt.write_map(self.Imat_dict[map_choice],self.meta['directory'])
                            
                    print('Export Complete')
    
                        
                    
                                    
                exp_button = Tk.Button(master=ewin,text='Export Data',command=_export_now)
                exp_button.grid(row=4,column=0)
                
                def _quit_ewin():
                    ewin.quit()
                    ewin.destroy()
                
                quit_button_exp = Tk.Button(master=ewin,text='Quit',command=_quit_ewin)
                quit_button_exp.grid(row=4,column=1)
                ewin.mainloop()
                
                
                
                
            
                
            export_button = Tk.Button(master=self.root,text="Export",command=_exp_win)
            export_button.grid(row=5,column=6)
            
            quit_button = Tk.Button(master=self.root,text="Quit",command=_quit_win)
            quit_button.grid(row=5,column=8)
    
    
    
            Tk.mainloop()
            
            
    def xyz(X,Y,Z):
         xnow = sum([complex(xi) for xi in X.split('+')])
         ynow = sum([complex(yi) for yi in Y.split('+')])
         znow = sum([complex(zi) for zi in Z.split('+')])
         xr,xi,yr,yi,zr,zi = float(np.real(xnow)),float(np.imag(xnow)),float(np.real(ynow)),float(np.imag(ynow)),float(np.real(znow)),float(np.imag(znow))
         return np.array([xr+1.0j*xi,yr+1.0j*yi,zr+1.0j*zi])/np.linalg.norm(np.array([xr+1.0j*xi,yr+1.0j*yi,zr+1.0j*zi]))

       
