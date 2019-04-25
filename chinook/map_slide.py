# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 16:28:47 2019

@author: rday
"""

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib import gridspec
import numpy as np


class window:

    def __init__(self,x,y,z,imap):
        self.x = x
        self.y = y
        self.z = z
        self.imap = imap
        self.make_window()
    
        
    def make_window(self):
        self.fig = plt.figure(figsize=(4,4))
    
        self.ax = plt.subplot2grid((10,10),(0,0),rowspan=9,colspan=10,fig=self.fig)
        
        self.axs = plt.subplot2grid((10,10),(9,0),rowspan=1,colspan=10,fig=self.fig)
        self.index = int(len(self.z)/2)
    
        self.img = self.ax.imshow(self.imap[:,:,self.index],extent=[self.x.min(),self.x.max(),self.y.min(),self.y.max()])    
        self.slide_obj = Slider(self.axs,'',self.z[0],self.z[-1],valinit=self.z[self.index],valstep=(self.z[1]-self.z[0]))
    
        self.ax.set_aspect((self.x.max()-self.x.min())/(self.y.max()-self.y.min()))
        
        def do_slide(val):
        
            self.index = int((val-self.z[0])/(self.z[1]-self.z[0]))

            self.img.set_data(self.imap[:,:,self.index])
            self.img.set_clim(vmin = self.imap[:,:,self.index].min()*0.8,vmax=self.imap[:,:,self.index].max()*1.2)
            self.fig.canvas.draw()
        
        self.slide_obj.on_changed(do_slide)
        plt.tight_layout()
        plt.show()
    

def dummy():
    x = np.linspace(-1,1,100)
    y = x.copy()
    z = np.linspace(1,10,100)
    X,Y = np.meshgrid(x,y)
    
    imap = np.zeros((len(x),len(y),len(z)))
    for ii in range(len(z)):
        imap[:,:,ii] = np.cos(z[ii]*(X**2+Y**2))
    
    return x,y,z,imap



if __name__ == "__main__":
    
    x,y,z,imap = dummy()
    window(x,y,z,imap)