# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 10:12:00 2018

@author: rday
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class dataset:
    
    def __init__(self,fnm):
        self.origin = fnm
        self.x,self.y,self.data = self.load_dataset()
        
        
    def plot_data(self):
        
        x,y = np.meshgrid(self.x,self.y)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        p=ax.pcolormesh(y.T,x.T,self.data.T,cmap=cm.Greys,vmax=self.data.max()/3)
        return p,ax
        
    
        


    def load_dataset(self):
        x,y,data =[],[],[]
        lc = 0
        with open(self.origin,'r') as origin:
            for line in origin:
                if lc==0:
                    x = np.array([float(t) for t in line.split('\t')[1:]])
                else:
                    num_line = [float(t) for t in line.split('\t')]
                    y.append(num_line[0])
                    data.append(num_line[1:])
                lc+=1
        origin.close()
        
        return x,np.array(y),np.array(data)
    
    
    
if __name__ == "__main__":
    
    fnm = 'C:/Users/rday/Documents/Elettra/FeSe_polarizationdependence/f0060_kw.txt' #HT_LH
    fnm = 'C:/Users/rday/Documents/Elettra/FeSe_polarizationdependence/f0059_kw.txt' #HT_LV
    fnm = 'C:/Users/rday/Documents/Elettra/FeSe_polarizationdependence/f0097_kw.txt' #LT_LH
#    fnm = 'C:/Users/rday/Documents/Elettra/FeSe_polarizationdependence/f0095_kw.txt' #LT_LV
    
    data = dataset(fnm)
    p,ax = data.plot_data()