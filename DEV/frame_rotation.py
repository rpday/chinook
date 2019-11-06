#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 14:03:35 2019

@author: ryanday
"""

import chinook.rotation_lib as rotlib

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np


deg = np.pi/180
rad = 180/np.pi



'''
Objective: for standard configuration, define generic rotation of sample.

Standard configuration involves: rotation about GLOBAL, FIXED vertical axis (y-axis),
and rotation about LOCAL horizontal axis (x-axis). Azimuthal rotation should be fixed about
the LOCAL z-axis.

Sample can be mounted initially at an ARBITRARY angle. I will do this in the Euler
convention of z-y'-z", but this can be updated later maybe to a more convenient convention.



'''

#def sample_euler(alpha,beta,gamma):
    
#    '''
#    Sample orientation, as mounted on sample holder defined by
#    3 Euler angles, following the z-y'-z" convention. Define here
#    the associated rotation matrix.
#    
#    '''
    
#    rot_mat = 
    
#def build_rotation_matrix(angles):
    

def cryo_polar(angle,frame):
    
    '''
    Rotate sample around the vertical polar angle. This axis is fixed to the 
    global coordinate frame and does not rotate with the sample
    
    *args*:
        
        - **angle**: float, angle in radians
        
        - **frame**: numpy array of 5x3 float, local sample and motor axes
        
    *return*:
        
        - **rot_frame**: numpy array of 5x3 float, local sample and motor axes after rotation
    '''
    
    axis = frame[4]
    rot_frame = def_plane(axis,angle,frame)
    
    return rot_frame
    

def cryo_tilt(angle,frame):
    
        
    '''
    Rotate sample around the horizontal tilt angle. This axis is fixed to the 
    sample coordinate frame and rotates with the sample's polar angle. This
    angle does not rotate with the sample's azimuthal rotation
    
    *args*:
        
        - **angle**: float, angle in radians
        
        - **frame**: numpy array of 5x3 float, local sample and motor axes
        
    *return*:
        
        - **rot_frame**: numpy array of 5x3 float, local sample and motor axes after rotation
    '''
    
    axis = frame[3]
    rot_frame = def_plane(axis,angle,frame)
    rot_frame[4] = frame[4]
    
    return rot_frame


def cryo_azim(angle,frame):
        
    '''
    Rotate sample around the normal azimuthal angle. This axis is fixed to the 
    sample coordinate frame rotates with the sample
    
    *args*:
        
        - **angle**: float, angle in radians
        
        - **frame**: numpy array of 5x3 float, local sample and motor axes
        
    *return*:
        
        - **rot_frame**: numpy array of 5x3 float, local sample and motor axes after rotation
    '''
    
    axis = frame[2]
    rot_frame = def_plane(axis,angle,frame)
    
    rot_frame[3:] = frame[3:]
    
    return rot_frame


def def_plane(axis,angle,frame):
    
    '''
    Rotate a frame of reference about a designated axis, by specified angle.
    
    *args*:
        
        - **axis**: numpy array of 3 float, axis of rotation
        
        - **angle**: float, angle of rotation, in radians
        
        - **frame**: numpy array of 5x3 float, local axes (sample_x,sample_y,sample_z,cryo_tilt,cryo_polar)

    '''
    
    
    rotation_matrix = rotlib.Rodrigues_Rmat(axis,angle)
    
    rotated_frame = np.einsum('ij,kj->ki',rotation_matrix,frame)
    
    return rotated_frame


def plottable_surface(frame_x,frame_y,frame_z):
    
    points = np.array([frame_x + frame_y,frame_x-frame_y,-frame_x+frame_y,-frame_x-frame_y])
    
    cart_lines = np.array([[np.zeros(3),frame_x],[np.zeros(3),frame_y],[np.zeros(3),frame_z]])
    
    
    return points,cart_lines


def get_cryo(frame):
    
    cryo_axes = 2*np.array([[-frame[3],frame[3]],[-frame[4],frame[4]]])
    
    return cryo_axes


def build_experiment(axes):
    
    frame = np.zeros((5,3))
    frame[:3,:] = axes
    
    frame[3,:] = np.array([1,0,0]) #tilt angle
    frame[4,:] = np.array([0,1,0]) #polar angle
    
    return frame


def show_frame(frame):
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    points,ax_lines = plottable_surface(frame[0],frame[1],frame[2])
    
    cryo_axes = get_cryo(frame)
    
    ax.plot_trisurf(points[:,0],points[:,1],points[:,2])
    ax.plot(ax_lines[0,:,0],ax_lines[0,:,1],ax_lines[0,:,2],c='r',lw=2)
    ax.plot(ax_lines[1,:,0],ax_lines[1,:,1],ax_lines[1,:,2],c='g',lw=2)
    ax.plot(ax_lines[2,:,0],ax_lines[2,:,1],ax_lines[2,:,2],c='b',lw=2)
    
    ax.plot(cryo_axes[0,:,0],cryo_axes[0,:,1],cryo_axes[0,:,2],c='k',lw=1,linestyle='dashed')
    ax.plot(cryo_axes[1,:,0],cryo_axes[1,:,1],cryo_axes[1,:,2],c='k',lw=1,linestyle='dashed')
    
    ax.set_xlim(-2,2)
    ax.set_ylim(-2,2)
    ax.set_zlim(-2,2)
    
    ax.set_aspect(1)
    

    


if __name__ == "__main__":
    
    axes = np.array([[1,0,0],[0,1,0],[0,0,1]])
    axes[2] = np.cross(axes[0],axes[1])
    
    frame = build_experiment(axes)
    show_frame(frame)
    
    rot_axis = np.array([0,1,1])
    polar_angle = 20*deg
    tilt_angle = 40*deg
    azim_angle = -15*deg
    
    rot_frame = cryo_polar(polar_angle,frame)
    
    show_frame(rot_frame)
    
    rot_frame2 = cryo_tilt(tilt_angle,rot_frame)
    
    show_frame(rot_frame2)
    
    rot_frame3 = cryo_azim(azim_angle,rot_frame2)
    
    show_frame(rot_frame3)
    
    
#    prime_axes = def_plane(rot_axis,rot_angle,*axes)
#    
#    show_frame(prime_axes[0],prime_axes[1],prime_axes[2])
    
    




    
    
    