#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 15:08:00 2019

@author: ryanday
"""

import chinook.orbital_plotting as oplot


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.widgets import RadioButtons,Slider
from matplotlib import gridspec
import matplotlib.cm as cm



class interface:
    '''
    This interactive tool is intended for exploring the orbital structure associated
    with an ARPES simulation using chinook.
    The user can scan through the datacube in each dimension, and visualize the orbitally
    projected wavefunction, as it pertains to any of the states in the calculation.
    
    This uses matplotlib natively rather than alternative gui systems in python like Tkinter,
    which makes it a bit more reliable across platforms.
    '''    
    
    
    def __init__(self,experiment,plot_grain=10):
        '''
        Initialize the interface. If necessary, the ARPES intensity is calculated
        in advance. The system is initialized as a cut in constant momentum along
        the y-direction. 
        
        *args*:
            - **experiment**: ARPES experiment object, as defined in *chinook.ARPES_lib*
            
        *kwargs*:
            - **plot_grain**: int, fine texture of the angular mesh for orbital plotting.
            Sets number of points along theta and phi at which the orbital surfaces are computed.
        '''
        self.experiment = experiment
        self.orbital_plottable = oplot.wavefunction(self.experiment.TB.basis,self.experiment.TB.Evec[0,:,0])
        self.plot_grain = plot_grain
        self.slice_dict = {'Momentum X':1,'Momentum Y':0,'Energy':2}
        self.other_dims = {0:(2,1),1:(2,0),2:(0,1)}
        try:
            _,self.Imat = self.experiment.spectral()
        except AttributeError:
            print('Matrix Elements have not yet been calculated. Running experiment.datacube() now.')
            self.experiment.datacube()
            print('Matrix element calculation complete.')
            _,self.Imat = self.experiment.spectral()

        self.axes = [np.linspace(*self.experiment.cube[1]),np.linspace(*self.experiment.cube[0]),np.linspace(*self.experiment.cube[2])]
        self.state_coords = np.array([self.axes[1][self.experiment.pks[:,2].astype(int)],self.axes[0][self.experiment.pks[:,1].astype(int)],self.experiment.pks[:,3]])
        self.state_coords[2,:] = self.bin_energy() #energy is discretized according to the energy mesh values to allow for state selection from angular maps
        
        self.aspects = [(self.experiment.cube[2][1]-self.experiment.cube[2][0])/(self.experiment.cube[0][1]-self.experiment.cube[0][0]),
                        (self.experiment.cube[2][1]-self.experiment.cube[2][0])/(self.experiment.cube[1][1]-self.experiment.cube[1][0]),
                        (self.experiment.cube[0][1]-self.experiment.cube[0][0])/(self.experiment.cube[1][1]-self.experiment.cube[1][0])]

        self.extents = [[self.axes[2][0],self.axes[2][-1],self.axes[1][-1],self.axes[1][0]],
                         [self.axes[2][0],self.axes[2][-1],self.axes[0][-1],self.axes[0][0]],
                          [self.axes[0][0],self.axes[0][-1],self.axes[1][-1],self.axes[1][0]]]
        
        self.dim = 0
        self.indx = int(len(self.axes[0])/2)
        self.deltas = [self.axes[0][1]-self.axes[0][0],self.axes[1][1]-self.axes[1][0],self.axes[2][1]-self.axes[2][0]]
        self.run_gui()
    
    def run_gui(self):
        '''
        Execution of the matplotlib gui. The figure is initialized, along with all widgets and 
        chosen datasets. The user has access to both the slice of ARPES data plotted, in addition
        to the orbital projection plotted in upper right panel.
        
        '''
        
        self.fig = plt.figure(figsize=(10,6))
        self.fig.canvas.set_window_title('Chinook Orbital Mapper')
        self.ax1 = plt.subplot2grid((10,17),(0,0),rowspan=10,colspan=9,fig=self.fig)
        self.ax2 = plt.subplot2grid((10,17),(0,11),rowspan=5,colspan=5,projection='3d',fig=self.fig)
        self.ax3 = plt.subplot2grid((10,17),(6,11),rowspan=1,colspan=5,fig=self.fig)
        self.ax4 = plt.subplot2grid((10,17),(7,11),rowspan=3,colspan=5,fig=self.fig)        

        self.img = self.ax1.imshow(self.Imat[self.indx,:,:],cmap=cm.magma,extent=self.extents[self.dim])
        
        
        self.ax1.set_aspect(self.aspects[self.dim])
        
        self.slide_w = Slider(self.ax3,'',self.axes[0][0],self.axes[0][-1],valinit=self.axes[0][self.indx],valstep =self.deltas[self.dim])
        
        self.radio_ax = RadioButtons(self.ax4,('Momentum X','Momentum Y','Energy'),active=1)
        
        
        ## INITIALIZE THE ORBITAL PLOT ##
        
        ind1,ind2 = self.other_dims[self.dim]
        img_centre = np.array([self.axes[ind1][int(len(self.axes[ind1])/2)],self.axes[ind2][int(len(self.axes[ind2])/2)]])
        self.cursor_index = self.find_cursor(img_centre)
        self.cursor, = self.ax1.plot(*self.state_coords[(ind1,ind2),self.cursor_index],marker='+',markersize=20,c='r')
        self.plot_peaks = self.state_coords[(ind1,ind2),:][:,np.where(self.state_coords[self.dim,:]==self.axes[self.dim][self.indx])[0]]
        self.bands, = self.ax1.plot(self.plot_peaks[0,:],self.plot_peaks[1,:],marker='o',linestyle='',markersize=2,c='w',alpha=0.1)
        self.orbital_plottable.vector = self.experiment.Ev[int(self.experiment.pks[self.cursor_index,0]/len(self.experiment.TB.basis)),:,int(self.experiment.pks[self.cursor_index,0]%len(self.experiment.TB.basis))]
        self.verts,self.triangles,self.colours = self.orbital_plottable.triangulate_wavefunction(self.plot_grain,plotting=False)
        self.orb_plot = self.orbital_plottable.plot_wavefunction(self.verts,self.triangles,self.colours,plot_ax = self.ax2)
        
        
        
        ## SLIDER FUNCTION ##
        def img_slide(val):
            '''
            User requests another slice of the image to be displayed using the 
            slider widget. Image is updated with the requested dataset.
            
            *args*:
                - **val**: float, slider value chosen
            
            '''
            self.indx = int((val-self.axes[self.dim][0])/(self.axes[self.dim][1]-self.axes[self.dim][0]))
            self.plot_img()
            self.fig.canvas.draw()
            plt.show()
            
            
        ## SLICE SELECTION RADIO BUTTON FUNCTION ##
        def button_click(label):
            '''
            User requests plotting of intensity map along an alternative axis.
            The intensity map is regenerated with axes scaled to reflect the 
            extent of data within the new plane.
            
            *args*:
                - **label**: string, radio-button label associated with the 
                possible slice directions
                
            '''
            self.dim = self.slice_dict[label]  
            self.indx = int(len(self.axes[self.dim])/2)
            
            if self.dim==0:
                self.img = self.ax1.imshow(self.Imat[self.indx,:,:],cmap=cm.magma,extent=self.extents[self.dim])
                self.img.set_clim(vmin = self.Imat[self.indx,:,:].min()*0.8,vmax=self.Imat[self.indx,:,:].max()*1.2)

            elif self.dim==1:
                self.img = self.ax1.imshow(self.Imat[:,self.indx,:],cmap=cm.magma,extent=self.extents[self.dim])
                self.img.set_clim(vmin = self.Imat[:,self.indx,:].min()*0.8,vmax=self.Imat[:,self.indx,:].max()*1.2)

            elif self.dim==2:
                self.indx = np.where(abs(self.axes[2])==abs(self.axes[2]).min())[0][0]
                self.img = self.ax1.imshow(self.Imat[:,:,self.indx],cmap=cm.magma,extent=self.extents[self.dim])
                self.img.set_clim(vmin = self.Imat[:,:,self.indx].min()*0.8,vmax=self.Imat[:,:,self.indx].max()*1.2)
            
            ind1,ind2 = self.other_dims[self.dim]
            self.plot_peaks = self.state_coords[(ind1,ind2),:][:,np.where(self.state_coords[self.dim,:]==self.axes[self.dim][self.indx])[0]]
            self.bands.set_data(self.plot_peaks[0,:],self.plot_peaks[1,:])
            
            self.slide_w.valmin = self.axes[self.dim][0]
            self.slide_w.valmax = self.axes[self.dim][-1]
            self.slide_w.delta = self.deltas[self.dim]
            self.slide_w.val = self.axes[self.dim][self.indx]
            self.ax1.set_aspect(self.aspects[self.dim])
  
            self.fig.canvas.draw()
        
        def onclick(event):
            '''
            User selects a point of interest in the intensity map, and the 
            corresponding orbital eigenvector is displayed in the upper right 
            panel of the window.
            
            *args*:
                - **event**: button_press_event associated with mouse query 
                within the frame of self.ax1. All other mouse events are
                disregarded.
            
            '''
            if event.inaxes==self.ax1.axes:
                coords = np.array([event.xdata,event.ydata])
                ind1,ind2 = self.other_dims[self.dim]
                
                self.cursor_index = self.find_cursor(coords)
                self.cursor.set_data(np.array(self.state_coords[(ind1,ind2),self.cursor_index]))
                
                psi = self.experiment.Ev[int(self.experiment.pks[self.cursor_index,0]/len(self.experiment.TB.basis)),:,int(self.experiment.pks[self.cursor_index,0]%len(self.experiment.TB.basis))]
                self.orbital_plottable.vector = psi
                
                self.verts,self.triangles,self.colours = self.orbital_plottable.triangulate_wavefunction(self.plot_grain,plotting=False)
                for o in range(len(self.orb_plot)):
                    self.orb_plot[o].remove()
                self.orb_plot = self.orbital_plottable.plot_wavefunction(self.verts,self.triangles,self.colours,plot_ax = self.ax2)
                self.fig.canvas.draw()
                plt.show()
                
                
        cid = self.fig.canvas.mpl_connect('button_press_event',onclick)
        self.slide_w.on_changed(img_slide)
        self.radio_ax.on_clicked(button_click)
        plt.tight_layout()
        plt.show()
        
    def find_cursor(self,cursor):
        '''
        Find nearest point to the desired cursor position, as clicked by the
        user. The cursor event coordinates are compared against the peak positions
        from the tight-binding calculation, and a best choice within the plotted 
        slice is selected.
        
        *args*:
            - **cursor**: tuple of 2 float, indicating the column and row of the
            event, in units of the data-set scaling (e.g. 1/A or eV)
        
        *return*:
            - **self.cursor_index**: tuple of int, indices of states associated with 
            the cursor event.
        '''
        ind1,ind2 = self.other_dims[self.dim]
        
        avail_states = np.where(self.state_coords[self.dim,:]==self.axes[self.dim][self.indx])[0]
        distance = np.linalg.norm((self.state_coords[(ind1,ind2),:][:,avail_states].T-cursor),axis=1)
        self.cursor_index = avail_states[np.where(distance==distance.min())[0][0]]
        self.ax1.set_title('X: {:0.04f}, Y: {:0.04f}'.format(*self.state_coords[(ind1,ind2),self.cursor_index]))
        return self.cursor_index
        
        
    def plot_img(self):
        '''
        Update the plotted intensity map slice. The plotted bandstructure states are also displayed.
        
        '''
        ind1,ind2 = self.other_dims[self.dim]
        self.plot_peaks = self.state_coords[(ind1,ind2),:][:,np.where(self.state_coords[self.dim,:]==self.axes[self.dim][self.indx])[0]]
        self.bands.set_data(self.plot_peaks[0,:],self.plot_peaks[1,:])
        if self.dim ==0:
            self.img.set_data(self.Imat[self.indx,:,:])
            self.img.set_clim(vmin = self.Imat[self.indx,:,:].min()*0.8,vmax=self.Imat[self.indx,:,:].max()*1.2)

        elif self.dim == 1:
            self.img.set_data(self.Imat[:,self.indx,:])
            self.img.set_clim(vmin = self.Imat[:,self.indx,:].min()*0.8,vmax=self.Imat[:,self.indx,:].max()*1.2)

        elif self.dim == 2:
            self.img.set_data(self.Imat[:,:,self.indx])
            self.img.set_clim(vmin = self.Imat[:,:,self.indx].min()*0.8,vmax=self.Imat[:,:,self.indx].max()*1.2)
            
    def bin_energy(self):
        '''
        Translate the exact energy value for the band peaks into the discrete
        binning of the intensity map, to allow for cursor queries to be processed.
        
        *return*:
            - **coarse_pts**: numpy array of float, same lengths as *self.state_coords*,
            but sampled over a discrete grid.
        '''
        
        fine_pts = self.state_coords[2,:]
        coarse_pts = np.array([self.axes[2][int((fine_pts[ii]-self.axes[2][0])/(self.axes[2][1]-self.axes[2][0]))] for ii in range(len(fine_pts))])
        return coarse_pts


if __name__ == "__main__":
    plotimg = interface(experiment)
