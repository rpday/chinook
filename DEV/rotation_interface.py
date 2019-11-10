#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Created on Sun Nov  3 17:01:08 2019

#@author: ryanday
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

import chinook.rotation_lib as rotlib

import os
rcParams.update({'figure.autolayout':True})
#rc('font',**{'family':'serif','serif':['Palatino'],'size':12})
#rc('text',usetex = False) 

import numpy as np

from matplotlib.figure import Figure
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

try:
    import tkinter as Tk
    from tkinter import messagebox
    from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,NavigationToolbar2Tk)
    tk_found = True
except ModuleNotFoundError:
    print('tkinter not found, please load for interactive map plotting')
    tk_found = False
    
rad = np.pi/180.0
hb = 6.626*10**-34/(2*np.pi)
q = 1.602*10**-19
AA = 10.0**-10
me = 9.11*10**-31

class sample:
    '''
    The sample object carries with it all information regarding rotation matrices,
    angles, as well as plot functions
    
    '''
    def __init__(self,master,klimits=[-1,1,-1,1],photon_energy=21.2):
        '''
        Initialize the object
        
        *args*
        
            - **figs**: matplotlib figures
            
            - **fig_wins**: Tkinter figure canvas
            
            - **axs**: matplotlib axes
            
            - **lattice**: numpy array of 3x3 float, indicating either lattice vectors or Cartesian axes of sample frame
            
            - **klimits**: list of 4 float, indicating range of k of interest
            
            - **photon_energy**: float, photon energy in eV
        
        '''
        self.master = master
        self.lattice = master.lattice
        self.lightsource = master.lightsource
        self.experiment = master.experiment
        self.rotation_matrix = np.identity(3)
        self.frame = self.build_experiment()
        
        self.figs = master.fig
        self.fig_wins = master.fig_win
        self.axs = master.ax
        
        self.angles = master.angle_values
        
        self.Kpts = self.define_kmesh(-0.1)
        self.theta_x,self.theta_y = self.emission_angles()
        self.draw_frame()
        self.draw_emission()
        
        
    def define_kmesh(self,binding_energy):
        '''
        Define k-vectors associated with the experimental region of interest.
        If an ARPES experiment has been defined associated with the experiment,
        takes the k-mesh from there. Otherwise, fills a unit square, with He1-alpha
        photon energy
        
        
        *args*:
            
            - **binding_energy**: float, binding energy of interest)
    
        *return*:
            
            - **Kpts**: numpy array of Nx4 float, kx,ky,kz,kn, momentum vectors
        '''
        
        if self.experiment is None:
            klimits = [-1,1,-1,1]
            photon_energy = 21.2
            kx = np.linspace(klimits[0],klimits[1],20)
            ky = np.linspace(klimits[2],klimits[3],20)
                    
            Kx,Ky = np.meshgrid(kx,ky)
            Kn = np.sqrt(2*me/hb**2*(photon_energy)*q)*AA
        else:
            Kx,Ky = self.experiment.X,self.experiment.Y
            Kn = np.sqrt(2*me/hb**2*(self.experiment.hv-self.experiment.W+binding_energy)*q)*AA
            

        Kpts = np.zeros((np.shape(Kx)[0]*np.shape(Kx)[1],4))
        Kpts[:,0] = Kx.flatten()
        Kpts[:,1] = Ky.flatten()
        Kpts[:,2]  = Kn-np.linalg.norm(Kpts[:,:2],axis=1)
        Kpts[:,3] = np.linalg.norm(Kpts,axis=1)
        
        return Kpts        
        
    def emission_angles(self):
        '''
        Rotate emission vectors, transform into angles, in degrees
        
        *return*:
            
            - **theta_x**: emission angle, projected along x-axis
            
            - **theta_y**: emission angle, projected along y-axis
        '''
    
        Kpts_new = np.einsum('ij,kj->ki',self.rotation_matrix,self.Kpts[:,:3])    
        theta = np.arccos(Kpts_new[:,2]/self.Kpts[:,3])
        phi = np.arctan2(Kpts_new[:,1],Kpts_new[:,0])
        
        theta_x = theta*np.cos(phi)/rad
        theta_y = theta*np.sin(phi)/rad
        
        return theta_x,theta_y

    def plottable_surface(self):
    
        '''
        Defines relevant vectors for drawing sample surface, as well as pertinent vectors
        e.g. the Cartesian frame, cryostat rotation axes, photon incidence vector.
        
        *return*:
            
            -  **points**: numpy array of 4x3 float, coordinates of sample corners
            
            - **cart_lines**: numpy array of 3x2x3 float, endpoints of cartesian axes, with origin as common for all
            
            - **nvec_line**: numpy array of 2x3 float, photon incidence vector
        
        '''
        points = np.array([self.frame[0] + self.frame[1],self.frame[0]-self.frame[1],-self.frame[0]+self.frame[1],-self.frame[0]-self.frame[1]])
    
        cart_lines = np.array([[np.zeros(3),self.frame[0]],[np.zeros(3),self.frame[1]],[np.zeros(3),self.frame[2]]])
        
        cryo_axes = 2*np.array([[-self.frame[3],self.frame[3]],[-self.frame[4],self.frame[4]]])
        nvec_line = np.array([np.zeros(3),self.lightsource.nvec_global])
        return points,cart_lines,cryo_axes,nvec_line
        
    def draw_frame(self):
        '''
        Execute rebuild of the sample plotting axes, filling in the sample surface
        as well as all vectors of interest.
        
        '''
        points,ax_lines,cryo_axes,nvec_line = self.plottable_surface()
        
        self.axs[0].clear()
        
        self.axs[0].plot_trisurf(points[:,0],points[:,1],points[:,2])
        self.axs[0].plot(ax_lines[0,:,0],ax_lines[0,:,1],ax_lines[0,:,2],c='r',lw=2)
        self.axs[0].plot(ax_lines[1,:,0],ax_lines[1,:,1],ax_lines[1,:,2],c='g',lw=2)
        self.axs[0].plot(ax_lines[2,:,0],ax_lines[2,:,1],ax_lines[2,:,2],c='b',lw=2)
    
        self.axs[0].plot(cryo_axes[0,:,0],cryo_axes[0,:,1],cryo_axes[0,:,2],c='k',lw=1,linestyle='dashed')
        self.axs[0].plot(cryo_axes[1,:,0],cryo_axes[1,:,1],cryo_axes[1,:,2],c='k',lw=1,linestyle='dashed')
        
        self.axs[0].plot(nvec_line[:,0],nvec_line[:,1],nvec_line[:,2],c='#7f00ff',lw=1)
        
        self.axs[0].set_xlim(-2,2)
        self.axs[0].set_ylim(-2,2)
        self.axs[0].set_zlim(-2,2)
    
        self.axs[0].set_aspect(1)
        self.figs[0].canvas.draw()
        
    def draw_emission(self):
        
        '''
        Execute re-draw of ARPES intensity map. If experiment has not been
        defined, just draws in a default intensity map to fill in as dummy-data.
        
        
        '''
        
        
        self.theta_x,self.theta_y = self.emission_angles()
        
        self.axs[1].clear()
        if self.experiment is None:
            Xm,Ym = np.reshape(self.theta_x,(20,20)),np.reshape(self.theta_y,(20,20))
            self.axs[1].pcolormesh(Xm,Ym,Xm**2+Ym**2)

        else:
            self.experiment.pol = self.lightsource.pol
            _,Imap = self.experiment.spectral()
            Xm,Ym = np.reshape(self.theta_x,np.shape(self.experiment.X)),np.reshape(self.theta_y,np.shape(self.experiment.Y))
            self.axs[1].pcolormesh(Xm,Ym,Imap[:,:,self.master.plot_index],cmap=cm.Greys_r)
                
        self.axs[1].set_xlim(-25,25)
        self.axs[1].set_ylim(-25,25)
        
        self.figs[1].canvas.draw()
        
        
    def build_experiment(self):
        '''
        Define the requisite vectors which establish the coordinate frame
        of an experiment. This includes the 'lattice' or by default 
        Cartesian basis vectors, as well as the axes of cryostat rotation,
        and the lightsource incident vector. Note this frame gets redefined
        every time the angles are updated so that the full rotation can be 
        done (rather than applying an incremental rotation, which shouldn't
        even really save much time computationally). Accordingly, 
        the frame is defined in reference to the global experimental frame
        
        *return*:
            
            - **frame**: numpy array of 6x3 float, containing basis vectors [0,1,2], 
            as well as tilt axis [3], polar axis [4] and the lightsource incidence [5].
            
            
        
        '''
    
        frame = np.zeros((6,3))
        frame[:3,:] = self.lattice
    
        frame[3,:] = np.array([1,0,0]) #tilt angle
        frame[4,:] = np.array([0,1,0]) #polar angle
        frame[5,:] = self.lightsource.nvec_global
    
        return frame
    


    def execute_rotation(self,cryo):
        
        '''
        Rotate sample around the normal azimuthal angle. This axis is fixed to the 
        sample coordinate frame rotates with the sample
        
        *args*:
            
            - **rotation_matrix**: numpy array of 3x3 float, full rotation matrix
            
            - **cryo**: numpy array of 3x3 float, cryostat (tilt, polar) rotation
            
        *return*:
            
            - **rot_frame**: numpy array of 5x3 float, local sample and motor axes after rotation
        
        fixed indicates which elements of frame do not rotate
        For example, for sample, fixed = (3,4)
        for polar, fixed = 3 no axes fixed
        for tilt, fixed = 4 only the tilt axis is fixed
        for azimuthal, (3,4) fixed
        
        '''
        frame = self.build_experiment()
        rotated_frame = np.einsum('ij,kj->ki',self.rotation_matrix,frame)
        frame[4] = np.array([0,1,0])
        frame[3] = np.dot(cryo,np.array([1,0,0]))
        return rotated_frame
    
    
    
    def rotate_frame(self):
        
        '''
        Define a single rotation matrix to account for the entire rotation scheme
        Order chosen to simplify best possible the definition.
        
        First, sample offset rotation is taken care of using the 3 Euler angles.
        
        Second, rotation around the cryostat tilt is done--this axis rotates with 
        the polar angle, so easiest to do this one first
        
        Third, polar angle is rotated
        
        Finally, azimuthal rotation of sample about the sample-holder normal is 
        carried out. This axis is rotated only through the polar and tilt angles
        
        Function directly modifies the system 'frame' as well as the 'rotation_matrix'
        
        '''
        
        angle_values = [ai*rad for ai in self.angles]
        euler = rotlib.Euler_to_R(*angle_values[:3])
        tilt = rotlib.Rodrigues_Rmat(np.array([1,0,0]),angle_values[4])
        polar = rotlib.Rodrigues_Rmat(np.array([0,1,0]),angle_values[3])
        
        cryo = np.dot(polar,tilt)
        
        az_prime = np.dot(cryo,np.array([0,0,1]))
        az_mat = rotlib.Rodrigues_Rmat(az_prime,angle_values[-1])
        
        self.rotation_matrix = np.dot(az_mat,np.dot(cryo,euler))
        
        self.frame = self.execute_rotation(cryo)
        
        self.lightsource.update(self.rotation_matrix)

        
class angle:
    '''
    Framework for widgets and control of sample orientation. Identical 
    copies generated for each of 6 angles available to user    
    
    Each angle has a label, as well as a text-entry box, linked to a
    slider for coarse and precise control 
    
    TODO: link change of entry-box to slider position
    
    *args*:
        
        - **name**: string, angle name, displayed in control panel
        
        - **index**: int, index of angle in list of angles
        
        - **value**: float, degrees initial angle value
        
        - **origin**: tuple of int, position of widgets row, column
        
        - **master**: Tkinter Frame root
        
        - **valmin**: float, minimum angle range
        
        - **valmax**: float, maximum angle range
    '''
    def __init__(self,name,index,value,origin,master,valmin=0,valmax=360,sample=None):
        self.master = master
        self.sample = sample
        self.index = index
        
        self.value = value
        self.strvar = Tk.StringVar()
        self.strvar.set('{:0.2f}'.format(value))
        self.label = Tk.Label(master=master,text=name).grid(row=origin[0],column=origin[1])
        
        self.entry = Tk.Entry(master=master,textvariable=self.strvar)
        self.entry.grid(row=origin[0],column=origin[1]+1)
        
        self.slider = Tk.Scale(master=master,from_=valmin,to=valmax,orient='horizontal',relief='flat',sliderrelief='flat',command=self.update_angle_slide,length=190)
        self.slider.grid(row=origin[0]+1,column=origin[1]+1)
    
    def update_angle_slide(self,event):
        '''
        Send updated slider value for parent angle to to execute reorientation
        and drawing of plot windows
        '''

        angle_value = self.slider.get()
        self.strvar.set('{:0.2f}'.format(angle_value))
        self.sample.angles[self.index] = angle_value
        if self.sample is not None:
            self.sample.rotate_frame()
            self.sample.draw_frame()
            self.sample.draw_emission()
    
    def update_angle_entry(self):
        '''
        Update entry in textbox for parent angle. Executes recalculation
        of the sample orientation and drawing of plots.
        '''
        angle_value = float(self.strvar.get())
        self.sample.angles[self.index] = angle_value
        self.slider.set(angle_value)
        if self.sample is not None:
            self.sample.rotate_frame()
            self.sample.draw_frame()
            self.sample.draw_emission()
            
            
class light:
    '''
    Light-vector object. Carries information on angle and orientation
    of incidence of light, as well as polarization. 
    
    Independent incidence vectors are maintained with respect to 
    sample and wrt laboratory frame: the light-vector controlled by user
    is not going to be the same as the one seen by the user. We draw the lab
    frame vector, but the sample reference frame vector is used for calculating
    the photoemission spectrum.
    
    '''
    
    
    def __init__(self):
        
        self.nvec = np.array([0,0,1]) #as seen by sample
        self.nvec_global = np.array([0,0,1]) # as seen by lab frame
        self.hpol = np.array([1,0,0])
        self.vpol = np.array([0,1,0])
        self.theta = 0
        self.phi = 0
        self.pol_type = 'LV'
        self.pol = np.array([0,1,0])
        
    def pol_update(self):
        '''
        Change the polarization type. Options are restricted
        to linear vertical and horizontal, as well as circular plus,
        circular minus, defined with respect to the incidence vector 
        direction
        '''
        if self.pol_type=='LH':
            self.pol = self.hpol
        elif self.pol_type == 'LV':
            self.pol = self.vpol
        elif self.pol_type == 'C+':
            self.pol = np.sqrt(0.5)*(self.hpol+1.0j*self.vpol)
        elif self.pol_type == 'C-':
            self.pol = np.sqrt(0.5)*(self.hpol-1.0j*self.vpol)
        
    
    def update(self,sample_rotation=None):
        '''
        Update the light definition. Redefine the basis polarization
        vectors (LV and LH) as indicated by the angles theta, phi
        defined as polar and azimuthal angles. If sample has also been
        rotated, the polarization with respect to sample is rotated
        further, by the inverse of the sample's rotation.
        
        *kwargs*:
            
            - **sample_rotation**: numpy array of 3x3 float, sample rotation
            matrix
        
        '''
        
        rotmat = np.dot(rotlib.Rodrigues_Rmat(np.array([0,1,0]),self.theta*rad),rotlib.Rodrigues_Rmat(np.array([0,0,1]),self.phi*rad))
        
        self.nvec_global = np.dot(rotmat,np.array([0,0,1]))
        
        if sample_rotation is not None:
            rotmat = np.dot(sample_rotation.T,rotmat)
        self.nvec = np.dot(rotmat,np.array([0,0,1]))
        
        self.hpol = np.dot(rotmat,np.array([1,0,0]))
        self.vpol = np.dot(rotmat,np.array([0,1,0]))        
        self.pol_update()
        
class Application:
    
    def __init__(self,experiment=None):
        
        self.lattice = np.identity(3)
        self.plot_index = 0
        self.experiment = experiment
        self.lightsource = light()
        self.root = Tk.Tk()
        self.fig = []
        self.ax = []
        self.fig_win = []
        if experiment is not None:
            self.energy_domain = np.linspace(*self.experiment.cube[2])
        else:
            self.energy_domain = np.linspace(-1,1,20)
        

        self.angle_values = np.zeros(6)
        self.angle_widgets = []
        
        self._define_message()
        
        self.root.wm_title('CHINOOK ORIENTATION')        
        
        self.make_panel()
        
        
    def _define_message(self):
        
        self.about_message ='This interactive program plots the experiment\n'
        self.about_message+='orientation along with real-time calculations\n'
        self.about_message+='of the photoemission spectra. The sample ori-\n'
        self.about_message+='entation is defined by the Euler angles alpha\n'
        self.about_message+='beta, gamma in the z-y-z convention. The\n'
        self.about_message+='cryostat orientation is defined with a verti-\n'
        self.about_message+='cal polar angle theta as well as tilt angle\n'
        self.about_message+='phi (axis rotates with theta) and azimuthal\n'
        self.about_message+='delta. The tilt and polar axes are drawn in\n'
        self.about_message+='dashed black lines. The light incidence\n'
        self.about_message+='vector is defined by a polar theta and azim-\n'
        self.about_message+='uthal phi. This vector is drawn in violet.\n'
        self.about_message+='The polarization can be chosen from linear\n'
        self.about_message+='vertical and horizontal, as well as circular\n'
        self.about_message+='plus and minus. Updates to text entries and\n'
        self.about_message+='polarization are effective only by pressing\n'
        self.about_message+='the *update* button. Plot axis ranges can be\n'
        self.about_message+='adjusted under the *settings*.'
        
    def _update(self):
        '''
        Update button forces execution of sample re-orientation
        and calculation of spectra accordingly. 
        
        '''
        
        for ai in self.angle_widgets:
            ai.update_angle_entry()
            
        self._update_polarization()
        self.sample.rotate_frame()
        self.sample.draw_frame()
        self.sample.draw_emission()
        
    def _update_polarization(self):
        self.lightsource.theta,self.lightsource.phi = self.nang[0].get(),self.nang[1].get()
        self.lightsource.pol_type = self.pol_labels[self.polChoice.get()] 
        self.lightsource.update()
        
                
    def _update_energy(self,event):
        binding_energy = self.energyslider.get()
        self.plot_index = np.where(abs(self.energy_domain-binding_energy)==abs(self.energy_domain-binding_energy).min())[0][0]
        self.sample.Kpts = self.sample.define_kmesh(binding_energy=binding_energy)
        self.sample.draw_emission()
        
    def _update_settings(self):
        print('update your settings')
        
    def _about_popup(self):
        
        top = Tk.Toplevel(self.root)
        top.title('About Chinook Orientation')
        
        msg = Tk.Message(master=top, text=self.about_message)
        msg.pack()
        
        closebutton = Tk.Button(master=top,text='Close',command=top.destroy)
        closebutton.pack()

#############################################################
#####BUILD ALL REQUISITE WIDGETS AND ASSOCIATED OBJECTS######
#############################################################        
        
    def _make_sample(self):
        crystal = sample(self)
        for ai in range(6):
            self.angle_widgets[ai].sample = crystal
        return crystal
        
    def _make_angles(self):
        names = ['alpha','beta','gamma','theta','phi','delta']
        origins = [(5,0),(8,0),(11,0),(5,4),(8,4),(11,4)]

        angle_range = [(0,180),(-90,90),(0,180),(-90,90),(-90,90),(-180,180)]
        
        for ii in range(6):
            self.angle_widgets.append(angle(name=names[ii],index=ii,value=self.angle_values[ii],origin=origins[ii],master=self.root,valmin=angle_range[ii][0],valmax=angle_range[ii][1]))  
    
    def _make_quit(self,origin):
        self.quit = Tk.Button(master=self.root,text="QUIT", command=self.root.destroy)
        self.quit.grid(row=origin[0],column=origin[1])
        
    def _make_update(self,origin):
        
        self.update = Tk.Button(master=self.root,text="UPDATE", command=self._update)
        self.update.grid(row=origin[0],column=origin[1])
        
    def _make_nvec(self,origin):
        
        self.nang = [Tk.DoubleVar(),Tk.DoubleVar()]
        self.nangEntry = []
        self.nang[0].set(0)
        self.nang[1].set(0)
        
        self.polButtons = []
        self.polChoice = Tk.IntVar()
        self.polChoice.set(0)
        self.pol_labels = ['LV','LH','C+','C-']
        
        columns = [1,4]
        button_columns = [0,1,4,5]
        
        
        self.nanglabel = Tk.Label(master=self.root,text='Photon Vector (θ,φ)').grid(row=origin[0]+1,column=origin[1])
        for ii in range(2):
            self.nangEntry.append(Tk.Entry(master=self.root,textvariable=self.nang[ii]))
            self.nangEntry[-1].grid(row=origin[0]+1,column=columns[ii])
            
        for ii in range(4):
            self.polButtons.append(Tk.Radiobutton(master=self.root,text=self.pol_labels[ii],relief='flat',indicatoron=0,variable=self.polChoice,value=ii,command=self._update_polarization))
            self.polButtons[-1].grid(row=origin[0],column=button_columns[ii])

    
    def _make_figure(self,origin,projection=None):
        
        self.fig.append(Figure(figsize=(4,4)))
        
        self.fig_win.append(FigureCanvasTkAgg(self.fig[-1],master=self.root))
        self.fig_win[-1].draw()
        self.ax.append(self.fig[-1].add_subplot(111,projection=projection))
        self.fig_win[-1].get_tk_widget().grid(row=origin[0],column=origin[1],columnspan=4,rowspan=4)

    def _make_energy(self,origin):
        
        self.energylabel = Tk.Label(master=self.root,text='Binding Energy (eV)').grid(row=origin[0],column=origin[1])
        self.energyslider = Tk.Scale(master=self.root,from_=self.energy_domain[0],to=self.energy_domain[-1],resolution=(self.energy_domain[1]-self.energy_domain[0]),orient='horizontal',relief='flat',sliderrelief='flat',command=self._update_energy)
        self.energyslider.grid(row=origin[0],column=origin[1]+1)
        
    def _make_settings(self,origin):
        
        self.settings_button = Tk.Button(master=self.root,text='Settings',command=self._update_settings)
        self.settings_button.grid(row=origin[0],column=origin[1])
        
    def _make_about(self,origin):
        
        self.about_button = Tk.Button(master=self.root,text='About',command=self._about_popup)
        self.about_button.grid(row=origin[0],column=origin[1])
        
        
        

    def make_panel(self): 
        
        self._make_figure(origin=(0,0),projection="3d")
        self._make_figure(origin=(0,5),projection=None)
        self._make_angles()
        self.sample = self._make_sample()
        self._make_nvec(origin=(14,0))
        self._make_energy(origin=(17,0))
        self._make_update(origin=(19,4))
        self._make_quit(origin=(19,5))
        self._make_settings(origin=(19,1))
        self._make_about(origin=(19,0))


if __name__ == "__main__":

    app = Application()#experiment=expmt)
    app.root.mainloop()
#root.destroy()


    
        
        