# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 09:51:27 2018
@author: rday
"""

import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter



def what_side(point,edge):
    '''
    The 'side' of a line edge point sits on, defined by the sign of the
    sgn function for the angle between the two, assuming a common origin.
    
    *args*:
        - **point**: numpy array of 2 float
        
        - **edge**: numpy array of 4 float
        
    *return*:
        +/-1 or 0 for sign of the cross product, which in 2D is a float, 
        equivalent to |point||edge|sin(THETA)
        
    ***
    '''
    return np.sign(np.around(np.cross(point,edge),3))

def crossing(point1,point2,vec,ezero):
    '''
    Find if the line defined by the points **point1** and **p2point2 cross the edge between
    vec[2:]-vec[:2].
    If the 'side' of the edge changes between **point1** and **point2**, then there is a crossing
    of edge between **point1** and **point2**.
    
    *args*:
        - **point1**, **point2**: numpy array of 2 float
    
        - **vec**: numpy array of 4 float, defining the 2 endpoints of the edge
    
    *return*:
        - int, +/-1,0. If sign changes, 1, else -1, and if we are on an unwanted edge, 
        throw the point away, flag as outside
    
    ***
    '''
    
    side1 = what_side((point1-vec[:2]),vec[2:]-vec[:2])
    side2 = what_side((point2-vec[:2]),vec[2:]-vec[:2])
    if side1!=side2:
        
        if side1==0 or side2==0:
            if ezero:

                return 1
            else:
                return -1
        else:
            return 1
    elif side1==side2 and side1!=0:
        return 0
    elif side1==side2 and side1==0:
        return 0
    
def edge_cross(point1,point2,edge2,edge1):
    
    '''
    Find if the point where the two lines cross: that defined by our point 
    and its automated endpoint, and the linesegment forming the side of
    the parallelogram.
    
    *args*:
        - **point1**, **point2**: length 2 numpy arrays of float defining
        the endpoints of line associated with point of interest
        
        - **edge1**, **edge2**: length 2 numpy arrays of float defining
        the endpoints of line associated with the parallelogram edge
        
    *return*:
        - if the two do indeed cross along the length of edge1-->edge2, 
        then return True, else False
    '''
    slope_e = (edge2-edge1)/np.linalg.norm(edge2-edge1)
    slope_p = (point2-point1)/np.linalg.norm(point2-point1)

    try:
        B = (slope_e[0]*(point1[1]-edge1[1]) - slope_e[1]*(point1[0]-edge1[0]))/(slope_p[0]*slope_e[1]-slope_p[1]*slope_e[0])
        inter = point1 + B*slope_p
        
    except ZeroDivisionError:
        print('Help--division by zero!')
        return False
    if 0<=np.dot(inter-edge1,slope_e)<np.linalg.norm(edge2-edge1):
        return True
    else:
        return False
    
def in_pgram(point2D,edge2D,origin,maxlen):
    
    '''
    Is a point p2D inside the parallelogram defined by e2D edges? 
    Create a line-segment beginning at p2D and extend along a length maxlen
    in length. If this line crosses an even number of edges,
    then it is outside, else inside
    
    *args*:
        
        - **point2D**: numpy array of 2 float
        
        - **edge2D**: numpy array of 4x4 float defining the endpoints of the edges in 2D
        
        - **origin**: numpy array of 2 integers, indicating which edges are the origin edges.
        Any points on an edge not in this set is outside!
        
        - **maxlen**: length of the line-segment, must be larger than the region of interest!
    
    *return*:
        - 1 if point2D is inside edge2D, 0 False, -1 if it is on a bad edge
        (terminate the search)
        
    ***
    '''
    point2D2 = np.around(point2D + np.array([277,331])*maxlen,4)    
    point2D = np.around(point2D,4)

    crossings = 0
    for e in list(enumerate(edge2D)):
        if (float(e[0]) in origin):
            origin_edge = True
        else:
            origin_edge = False
        cross_edge = crossing(point2D,point2D2,e[1],origin_edge)
        if cross_edge<0:
            return -1
        elif cross_edge>0:
            boolc = edge_cross(point2D,point2D2,e[1][:2],e[1][2:])
            if boolc:
                crossings+=1
    if np.mod(crossings,2)==0:
        return 0
    else:
        return 1
    
def plot(pts,inside_pt,outside_pt,edges):
    '''
    Diagnostic function, for plotting a defined parallelogram, alongside
    a set of points, coloured corresponding to whether or not they are indeed
    inside the shape, or not.
    
    *args*:
        - **pts**: numpy array of Nx2 dimensional floats, points to check
        for in/out condition 
        
        - **inside**, **outside**: numpy arrays of coordinates from **pts**
        corresponding to those inside and outside the shape, respectively
        
        - **edges**: numpy array of 4x4 float indicating coordinates of the
        shape's corners
        
    *return*:
        - **ax**: matplotlib axes object
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.scatter(pts[:,0],pts[:,1],c='b')
    
    for i in range(4):
        
        plot_pts = np.reshape(edges[np.mod(i,4)],(2,2))
        ax.plot(plot_pts[:,0],plot_pts[:,1],c='b')
        
    ax.scatter(inside_pt[:,0],inside_pt[:,1],c='r')
    ax.scatter(outside_pt[:,0],outside_pt[:,1],c='k')
    return ax
    
def check_in(points,edges):
    '''
    Check a set of points for the condition of being inside the shape defined by edges.
    
    *args*:
        - **points**: numpy array of Nx2 float
        
        - **edges**: numpy array of 4x4 float (edges of shape)
    
    *return*:
        - **inside**, **outside**: numpy arrays of Yx2 and (N-Y)x2 float respectively
        
    ***
    '''
    pts_inside = []
    pts_outside = []
    for p in points:
        if in_pgram(p,edges,5):
            pts_inside.append([p[0],p[1]])
        else:
            pts_outside.append([p[0],p[1]])
    if len(pts_inside)==0:
        print('No points inside this form!')
    return np.array(pts_inside),np.array(pts_outside)

def sort_pts(pts):
    '''
    Sort the edges corresponding to their order, such that continuous,
    convex edges can be defined.
    Define a centre of mass coordinate as origin, and then find the angle
    the points make to this common origin. In this way, a simple ordering 
    based on the angle can be defined.
    
    *args*:
        - **pts**: numpy array of 4x2 float
        
    *return*:
        - sorted points numpy array of 4x3 float
        
    ***
    '''
    com = np.sum(pts,0)*0.25
    pt_s = np.zeros((4,np.shape(pts)[1]+1))
    pt_s[:,:2] = pts
    pt_s[:,2] = np.mod(np.arctan2(pt_s[:,1]-com[1],pt_s[:,0]-com[0])*180/np.pi,360)
    pt_s = sorted(pt_s,key=itemgetter(2))
    return np.array(pt_s)[:,:2]
    
