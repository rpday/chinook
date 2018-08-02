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
    The 'side' of a line edge point sits on, defined by the sign of the sin function for the angle between the two, assuming a common origin.
    args:
        point --numpy array of 2 float
        edge -- numpy array of 4 float
        
    return:
        +/-1 or 0 for sign of the cross product, which in 2D is a float, equivalent to |point||edge|sin(THETA)
    '''
    return np.sign(np.cross(point,edge))

def crossing(p1,p2,e):
    '''
    Find if the line defined by the points p1 and p2 cross the line e. If the 'side' of e changes between p1 and p2, then there is a crossing point of e
    between p1 and p2/
    args:
        p1,p2 -- numpy array of 2 float
        e -- numpy array of 4 float
    return:
        Boolean: if sign changes, True, else False
    '''
    
    s1 = what_side((p1-e[:2]),e[:2]-e[2:])
    s2 = what_side((p2-e[:2]),e[:2]-e[2:])
    if s1!=s2:
        if s1==0 or s2==0:
            return True
        else:
            return True
    elif s1==s2 and s1!=0:
        return False
    elif s1==s2 and s1==0:
        return False
    
def edge_cross(p1,p2,e2,e1):
    '''
    Find if the point where the two lines cross: that defined by our point and its automated endpoint, and the linesegment 
    forming the side of the parallelogram.
    args:
        p1,p2,e2,e1: length 2 numpy arrays of float defining the endpoints of the two lines
    return:
        if the two do indeed cross along the length of e1-->e2, then return True, else False
    '''
    me = (e2-e1)/np.linalg.norm(e2-e1)
    mp = (p2-p1)/np.linalg.norm(p2-p1)

    try:
        B = (me[0]*(p1[1]-e1[1]) - me[1]*(p1[0]-e1[0]))/(mp[0]*me[1]-mp[1]*me[0])
        inter = p1 + B*mp
        
    except ZeroDivisionError:
        print('Help--division by zero!')
        return False
    if 0<=np.dot(inter-e1,me)<np.dot(e2-e1,me):
        return True
    else:
        return False
    
def in_pgram(p2D,e2D,maxlen):
    '''
    Is a point p2D inside the parallelogram defined by e2D edges? Create a line-segment beginning
    at p2D and extend along a length maxlen in length. If this line crosses an even number of edges,
    then it is outside, else inside
    args:
        p2D -- numpy array of 2 float
        e2D -- numpy array of 4x4 float defining the endpoints of the edges in 2D
        maxlen -- length of the line-segment -- must be larger than the region of interest!
    return:
        Boolean True if p2D is inside e2D, else False
    '''
    
    p2D2 = p2D + np.array([277,331])*maxlen
    p2D = np.around(p2D,4)
    crossings = 0

    for e in e2D:
        cross_edge = crossing(p2D,p2D2,e)
        if cross_edge:
            boolc = edge_cross(p2D,p2D2,e[:2],e[2:])
            if boolc:
                crossings+=1
    if np.mod(crossings,2)==0:
        return False
    else:
        return True
    
def plot(pts,inside,outside,edges):
    '''
    Diagnostic function, for plotting a defined parallelogram, alongside a set of points, coloured corresponding
    to whether or not they are indeed inside the shape, or not
    args:
        pts -- points to check for in/out condition (numpy array of Nx2 dimensional floats)
        inside,outside -- numpy arrays of coordinates from pts corresponding to those inside and outside the shape, respectively
        edges -- numpy array of 4x4 float indicating coordinates of the shape's corners
    return None
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.scatter(pts[:,0],pts[:,1],c='b')
    for i in range(4):
        tmp = np.reshape(edges[np.mod(i,4)],(2,2))
        ax.plot(tmp[:,0],tmp[:,1],c='b')
    ax.scatter(inside[:,0],inside[:,1],c='r')
    ax.scatter(outside[:,0],outside[:,1],c='k')
    return None
    
def check_in(points,edges):
    '''
    Check a set of points for the condition of being inside the shape defined by edges.
    args:
        points -- numpy array of Nx2 float
        edges -- numpy array of 4x4 float (edges of shape)
    return:
        inside,outside -- numpy arrays of Yx2 and (N-Y)x2 float respectively
    '''
    inside = []
    outside = []
    for p in points:
        if in_pgram(p,edges,5):
            inside.append([p[0],p[1]])
        else:
            outside.append([p[0],p[1]])
    if len(inside)==0:
        print('No points inside this form!')
    return np.array(inside),np.array(outside)

def sort_pts(pts):
    '''
    Sort the edges corresponding to their order, such that continuous, convex edges can be defined.
    Define a centre of mass coordinate as origin, and then find the angle the points make to this common
    origin. In this way, a simple ordering based on the angle can be defined.
    args:
        pts -- numpy array of 4x2 float
    return:
        sorted points numpy array of 4x3 float
    '''
    com = np.sum(pts,0)*0.25
    pt_s = np.zeros((4,np.shape(pts)[1]+1))
    pt_s[:,:2] = pts
    pt_s[:,2] = np.mod(np.arctan2(pt_s[:,1]-com[1],pt_s[:,0]-com[0])*180/np.pi,360)
    pt_s = sorted(pt_s,key=itemgetter(2))
    return np.array(pt_s)[:,:2]
    
if __name__ == "__main__":
    
    a,b = np.array([2*np.random.random() for i in range(2)]),np.array([2*np.random.random() for i in range(2)])
    pts = sort_pts(np.array([np.zeros(2),a,a+b,b]))
    pts2 = np.array([[np.random.random()*3-.5,np.random.random()*3-0.5] for ii in range(2000)])
    edges = np.array([[*pts[np.mod(ii,4)],*pts[np.mod(ii+1,4)]] for ii in range(4)])
    
    inside,outside = check_in(pts2,edges)
    
    plot(pts,inside,outside,edges)