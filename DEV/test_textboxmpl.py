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
import os
rcParams.update({'figure.autolayout':True})
rc('font',**{'family':'serif','serif':['Palatino'],'size':12})
rc('text',usetex = False) 

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
    

    
    
    
    
class Application:
    
    def __init__(self,lattice):
        self.lattice = lattice
        self.root = Tk.Tk()
        self.fig = []
        self.ax = []
        self.fig_win = []
        
        self.root.wm_title('CHINOOK ORIENTATION')
        ### insert experiment object here
        
        
        self.make_panel()
    
    def _make_quit(self):
        self.quit = Tk.Button(master=self.root,text="QUIT", command=self.root.quit)

        self.quit.grid(row=13,column=3)

    def _make_figure(self,origin):
        
        self.fig.append(Figure(figsize=(4,4)))
        self.ax.append(self.fig[-1].add_subplot(111))
        
        self.fig_win.append(FigureCanvasTkAgg(self.fig[-1],master=self.root))
        self.fig_win[-1].show()
        self.fig_win[-1].get_tk_widget().grid(row=origin[0],column=origin[1],columnspan=4,rowspan=4)

    def make_panel(self): 
        
        self._make_figure(origin=(0,0))
        self._make_figure(origin=(0,4))
        self._make_quit()


if __name__ == "__main__":
    lattice = np.identity(3)

    app = Application(lattice=lattice)
    app.root.mainloop()
#root.destroy()


    
        
        