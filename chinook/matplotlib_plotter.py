# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 14:53:36 2019

@author: rday
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 15:08:00 2019

@author: ryanday
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.widgets import RadioButtons,Slider
from matplotlib import gridspec
import matplotlib.cm as cm



class interface:
    '''
    This interactive tool is intended for exploring the dataset associated
    with an ARPES simulation using chinook.
    The user can scan through the datacube in each dimension.
    This uses matplotlib natively rather than alternative gui systems in python like Tkinter,
    which makes it a bit more robust across platforms.
    '''    
    
    
    def __init__(self,experiment):
        '''
        Initialize the interface. If necessary, the ARPES intensity is calculated
        in advance. The system is initialized as a cut in constant momentum along
        the y-direction. 
        
        *args*:

            - **experiment**: ARPES experiment object, as defined in *chinook.ARPES_lib*
        
        ***
        '''
        self.experiment = experiment
        self.slice_dict = {'Fixed Kx':1,'Fixed Ky':0,'Fixed Energy':2}
        self.other_dims = {0:(2,1),1:(2,0),2:(1,0)} #relevant axes for rows, columns of each dataset
        try:
            _,self.Imat = self.experiment.spectral()
        except AttributeError:
            print('Matrix Elements have not yet been calculated. Running experiment.datacube() now.')
            self.experiment.datacube()
            print('Matrix element calculation complete.')
            _,self.Imat = self.experiment.spectral()

        self.axes = [np.linspace(*self.experiment.cube[1]),np.linspace(*self.experiment.cube[0]),np.linspace(*self.experiment.cube[2])]
        self.state_coords = np.array([self.axes[0][self.experiment.pks[:,1].astype(int)],self.axes[1][self.experiment.pks[:,2].astype(int)],self.experiment.pks[:,3]]) ## Y, X , Energy coordinates of states
        self.state_coords[2,:] = self.bin_energy() #energy is discretized according to the energy mesh values to allow for state selection from angular maps

        self.aspects = [(self.experiment.cube[self.other_dims[ii][0]][1]-self.experiment.cube[self.other_dims[ii][0]][0])/(self.experiment.cube[self.other_dims[ii][1]][1]-self.experiment.cube[self.other_dims[ii][1]][0]) for ii in range(3)]        
        self.extents = [[self.axes[self.other_dims[ii][0]][0],self.axes[self.other_dims[ii][0]][-1],self.axes[self.other_dims[ii][1]][0],self.axes[self.other_dims[ii][1]][-1]] for ii in range(3)]
        self.deltas = [self.axes[0][1]-self.axes[0][0],self.axes[1][1]-self.axes[1][0],self.axes[2][1]-self.axes[2][0]]
        
        self.run_gui()
    
    def run_gui(self):

        '''
        Execution of the matplotlib gui. The figure is initialized, along with all widgets and 
        chosen datasets. The user has access to both the slice of ARPES data plotted, in addition
        to the orbital projection plotted in upper right panel.
        
        '''
        
        self.fig = plt.figure(figsize=(11,11))
        self.fig.canvas.set_window_title('Chinook Orbital Mapper')
        self.ax1 = plt.subplot2grid((10,10),(0,0),rowspan=6,colspan=6,fig=self.fig) #main figure
        self.ax2 = plt.subplot2grid((10,10),(0,7),rowspan=6,colspan=3,fig=self.fig) #subfigure, right
        self.ax3 = plt.subplot2grid((10,10),(7,0),rowspan=3,colspan=6,fig=self.fig) #subfigure, bottom
        self.ax4 = plt.subplot2grid((10,10),(7,7),rowspan=1,colspan=3,fig=self.fig) #slider axis
        self.ax5 = plt.subplot2grid((10,10),(8,7),rowspan=3,colspan=3,fig=self.fig) #radio button axis

        self.dim = 0 #initialize looking at constant Y contour, midway through dataset
        self.indx = int(len(self.axes[0])/2)

        self.img = self.ax1.imshow(self.Imat[self.indx,:,:],cmap=cm.magma,extent=self.extents[self.dim],origin='lower')
        
        self.ax1.set_aspect(self.aspects[self.dim])
        
        self.slide_w = Slider(self.ax4,'',self.axes[0][0],self.axes[0][-1],valinit=self.axes[0][self.indx],valstep =self.deltas[self.dim],color='#1F1B33')
        
        self.radio_ax = RadioButtons(self.ax5,('Fixed Kx','Fixed Ky','Fixed Energy'),active=1,activecolor='#1F1B33')
        
        
        ## INITIALIZE THE ORBITAL PLOT ##
        
        ind1,ind2 = self.other_dims[self.dim]  ##rows, columns
        self.cursor_location =  np.array([self.axes[ind1][int(len(self.axes[ind1])/2)],self.axes[ind2][int(len(self.axes[ind2])/2)]]) ##row, column coordinates of centre, in phys. units
        self.rowdc, = self.ax2.plot(self.Imat[self.indx,:,int(len(self.axes[ind1])/2)],self.axes[ind2]) #all rows distribution curve, here fixed energy, all kx
        self.coldc, = self.ax3.plot(self.axes[ind1],self.Imat[self.indx,int(len(self.axes[ind2])/2),:]) #all column distribution curve, here fixed kx, all energy
        self.cursor, = self.ax1.plot([self.cursor_location[0]],[self.cursor_location[1]],marker='+',markersize=20,c='r')
        self.ax1.set_title('Cursor: E: {:0.04f}, X: {:0.04f}'.format(self.cursor_location[0],self.cursor_location[1]))

       

        self.plot_peaks = self.state_coords[(ind1,ind2),:][:,np.where(self.state_coords[self.dim,:]==self.axes[self.dim][self.indx])[0]]
        self.bands, = self.ax1.plot(self.plot_peaks[0,:],self.plot_peaks[1,:],marker='o',linestyle='',markersize=2,c='w',alpha=0.3)
        
        
        
        ## SLIDER FUNCTION ##
        def img_slide(val):

            '''
            User requests another slice of the image to be displayed using the 
            slider widget. Image is updated with the requested dataset.
            
            *args*:

                - **val**: float, slider value chosen

            ***
            '''
            self.indx = int((val-self.axes[self.dim][0])/(self.axes[self.dim][1]-self.axes[self.dim][0]))
            self.plot_img()
            self.find_cursor()
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

            ***    
            '''
            self.dim = self.slice_dict[label]  
            
            ind1,ind2 = self.other_dims[self.dim]
            self.cursor_location = np.array([self.axes[ind1][int(len(self.axes[ind1])/2)],self.axes[ind2][int(len(self.axes[ind2])/2)]]) #column, row in physical units
            
            self.indx = int(len(self.axes[self.dim])/2)
            if self.dim==0:
                
                self.img = self.ax1.imshow(self.Imat[self.indx,:,:],cmap=cm.magma,extent=self.extents[self.dim],origin='lower')
                self.img.set_clim(vmin = self.Imat[self.indx,:,:].min()*0.8,vmax=self.Imat[self.indx,:,:].max()*1.2)

            elif self.dim==1:
                self.img = self.ax1.imshow(self.Imat[:,self.indx,:],cmap=cm.magma,extent=self.extents[self.dim],origin='lower')
                self.img.set_clim(vmin = self.Imat[:,self.indx,:].min()*0.8,vmax=self.Imat[:,self.indx,:].max()*1.2)

            elif self.dim==2:
                self.img = self.ax1.imshow(self.Imat[:,:,self.indx],cmap=cm.magma,extent=self.extents[self.dim],origin='lower')
                self.img.set_clim(vmin = self.Imat[:,:,self.indx].min()*0.8,vmax=self.Imat[:,:,self.indx].max()*1.2)
            
            self.find_cursor()

            self.plot_peaks = self.state_coords[(ind1,ind2),:][:,np.where(self.state_coords[self.dim,:]==self.axes[self.dim][self.indx])[0]]
            self.bands.set_data(self.plot_peaks[0,:],self.plot_peaks[1,:])

            self.slide_w.valmin = self.axes[self.dim][0]
            self.slide_w.valmax = self.axes[self.dim][-1]
            self.slide_w.ax.set_xlim(self.slide_w.valmin,self.slide_w.valmax)
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

            ***
            '''
            if event.inaxes==self.ax1.axes:
                coords = np.array([event.xdata,event.ydata])
                ind1,ind2 = self.other_dims[self.dim]
                self.cursor_location = np.array([coords[0],coords[1]])
                self.find_cursor()

                self.fig.canvas.draw()
                plt.show()
                
                
        cid = self.fig.canvas.mpl_connect('button_press_event',onclick)
        self.slide_w.on_changed(img_slide)
        self.radio_ax.on_clicked(button_click)
        plt.tight_layout()
        plt.show()
        
    def find_cursor(self):

        '''
        Find nearest point to the desired cursor position, as clicked by the
        user. The cursor event coordinates are compared against the peak positions
        from the tight-binding calculation, and a best choice within the plotted 
        slice is selected.
        
        *args*:

            - **cursor**: tuple of 2 float, indicating the column and row of the
            event, in units of the data-set scaling (e.g. 1/A or eV)
        
        ***
        '''
        ind1,ind2 = self.other_dims[self.dim] #columns,rows
        cursor = self.cursor_location
        
        axis1 = self.axes[ind1] #
        axis2 = self.axes[ind2]

        row_index = np.where((abs(axis1-cursor[0])==abs(axis1-cursor[0]).min()))[0][0]
        col_index = np.where((abs(axis2-cursor[1])==abs(axis2-cursor[1]).min()))[0][0]
        
        if self.dim == 0:
            rowdc = self.Imat[self.indx,:,row_index]
            coldc = self.Imat[self.indx,col_index,:]
            imax = self.Imat[self.indx,:,:].max()
            self.ax1.set_title('Cursor: E: {:0.04f}, Kx: {:0.04f}'.format(cursor[0],cursor[1]))

        if self.dim == 1:
            rowdc = self.Imat[:,self.indx,row_index]
            coldc = self.Imat[col_index,self.indx,:]
            imax = self.Imat[:,self.indx,:].max()
            self.ax1.set_title('Cursor: E: {:0.04f}, Ky: {:0.04f}'.format(cursor[0],cursor[1]))

        if self.dim == 2:
            rowdc = self.Imat[:,row_index,self.indx]
            coldc = self.Imat[col_index,:,self.indx]
            imax = self.Imat[:,:,self.indx].max()
            self.ax1.set_title('Cursor: Kx: {:0.04f}, Ky: {:0.04f}'.format(cursor[0],cursor[1]))

        
        self.rowdc.set_data(rowdc,axis2)
        self.coldc.set_data(axis1,coldc)
        self.cursor.set_data([cursor[0]],[cursor[1]])
        self.ax2.set_ylim(axis2.min(),axis2.max())
        self.ax2.set_xlim(rowdc.min(),imax)
        self.ax3.set_ylim(coldc.min(),imax)
        self.ax3.set_xlim(axis1.min(),axis1.max())    

        
        
    def plot_img(self):

        '''
        Update the plotted intensity map slice. The plotted bandstructure states are 
        also displayed.
        
        ***
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

        ***
        '''
        
        fine_pts = self.state_coords[2,:]
        coarse_pts = np.array([self.axes[2][int((fine_pts[ii]-self.axes[2][0])/(self.axes[2][1]-self.axes[2][0]))] if (fine_pts[ii]<self.axes[2].max() and fine_pts[ii]>self.axes[2].min()) else -1 for ii in range(len(fine_pts))])
        return coarse_pts


if __name__ == "__main__":
    plotimg = interface(expmt)
