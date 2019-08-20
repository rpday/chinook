# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:08:44 2019

@author: rday
"""

import h5py
import matplotlib.pyplot as plt
import numpy as np



class arpes_dataset:
    
    
    def __init__(self,filename):
        
        self.meta,self.dataset = self.load_dataset(filename)
        
        
        
    
    
    def plot_data(self,slice_select=None):
        '''
        FILL IN FUNCTIONALITY FROM ARPES_lib.py for plotting slices of 3D datasets. This incidentally has also been updated to give
        more pythonic code.
        '''
        
        all_dims = np.array([self.meta['Momentum X (1/A)'],self.meta['Momentum Y (1/A)'],self.meta['Energy (eV)']])
        
        if slice_select is None:
            
            
            shape_arr = np.shape(self.dataset)
            if len(shape_arr)>2:
                print('Error, must specify a slice for 3D blocks of data before plotting')
            arr_1 = np.where(all_dims[:,-1]==shape_arr[0])[0][0]
            arr_2 = np.where(all_dims[:,-1]==shape_arr[1])[0][0]
            
        x = np.linspace(all_dims[arr_2,0],all_dims[arr_2,1],int(all_dims[arr_2,2]))
        y = np.linspace(all_dims[arr_1,0],all_dims[arr_1,1],int(all_dims[arr_1,2]))
            
        X,Y = np.meshgrid(x,y)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.pcolormesh(X,Y,self.dataset)
            
            
            
        
        
        
        
        
        
        
        
    def load_dataset(self,filename):
        '''
        Load dataset from file. This will open either text or hdf5 type data.
        
        *args*:
            - **filename**: string, location on disk of datafile
            
        *return*:
            - **dataset**: tuple including: metadata as dictionary, array of data
        
        '''
        extension = filename.split('.')[-1]
        
        if extension == 'hdf5':
            dataset = self.load_hdf5(filename)
            
        elif extension == 'txt':
            
            dataset = self.load_txt(filename)
        
        
        return dataset
        
    
    def load_hdf5(self,filename):
        
        with h5py.File(filename,'r') as my_file:
            group = list(my_file.keys())[0]
            dataset = list(my_file[group].keys())[0]
            full_path = '/'.join((group,dataset))
            data_arr = my_file[full_path]
            
            attributes = {ai:data_arr.attrs[ai] for ai in data_arr.attrs.keys()}
            data  = np.array(data_arr)
            
        return attributes,data
        
        
    
    
if __name__ == "__main__":
    
    filestring = 'C:/Users/rday/Documents/TB_ARPES/2019/Impurity_Model/12_08/fresh_imp/chinook_intensity.hdf5'
    
    my_data = arpes_dataset(filestring)