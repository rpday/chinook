# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 09:51:27 2018

@author: rday
"""

import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter


def plane_project(point,plane):
    norm = plane[:3]/np.linalg.norm(plane[:3])
    return norm*(np.dot(norm,(plane[3:6]-point))) + point
    


def rotation(norm):
    '''
    Rotate the plane so that it is normal to the 001 direction, then strip
    the z-information.
    '''
    if np.linalg.norm(norm-np.array([0,0,1]))!=0:
        x = np.cross(norm,np.array([0,0,1]))
        sin = np.linalg.norm(x)
        x = x/sin
        cos = np.dot(norm,np.array([0,0,1]))
        u = np.array([[0,-x[2],x[1]],[x[2],0,-x[0]],[-x[1],x[0],0]])
        uu = np.array([x[0]*x,x[1]*x,x[2]*x])
        R = cos*np.identity(3) + sin*u + (1-cos)*uu
    else:
        R = np.identity(3)
    return R

def what_side(point,edge):
    return np.sign(np.cross(point,edge))

def crossing(p1,p2,e):
    
    s1 = what_side((p1-e[:2]),e[:2]-e[2:])
    s2 = what_side((p2-e[:2]),e[:2]-e[2:])
    if s1!=s2:
        return True
    elif s1==s2 and s1!=0:
        return False
    elif s1==s2 and s1==0:
        return False
    
def edge_cross(p1,p2,e2,e1):
    me = (e2-e1)/np.linalg.norm(e2-e1)
    mp = (p2-p1)/np.linalg.norm(p2-p1)

    try:
#        inter = ((e1[0]-p1[0])*mp[1] + (p1[0]-e1[0])*mp[0])/(mp[0]*me[1]-me[0]*mp[1])*me+e1
        B = (me[0]*(p1[1]-e1[1]) - me[1]*(p1[0]-e1[0]))/(mp[0]*me[1]-mp[1]*me[0])
        inter = p1 + B*mp
        
    except ZeroDivisionError:
        print('Help--division by zero!')
        return False
    if 0<=np.dot(inter-e1,me)<np.dot(e2-e1,me):
#        print(inter)

        return True
    else:
        return False
    
def in_pgram(point,plane,e2D,maxlen):
    Rmat = rotation(plane[:3])
    p_proj = plane_project(point,plane)
    p2D = np.dot(p_proj,Rmat)[:2]
    p2D2 = p2D + (e2D[0,:2]-e2D[0,2:])*2*maxlen
   
    crossings = 0

    for e in e2D:

        if crossing(p2D,p2D2,e):
            boolc = edge_cross(p2D,p2D2,e[:2],e[2:])
            if boolc:
                crossings+=1
    if np.mod(crossings,2)==0:
        return False
    else:
        plt.figure()
        tmp = np.array(list(e2D) + [e2D[0]])
        plt.plot(tmp[:,0],tmp[:,1])
        plt.scatter(p2D[0],p2D[1])
        plt.scatter(p2D2[0],p2D[1])
        return True
    
    
def in_pgram2D(point,norm,e2D,maxlen):
    Rmat = rotation(norm)
    p2D = np.dot(point,Rmat)[:2]
    p2D2 = p2D + (e2D[0,:2]-e2D[0,2:])*2*maxlen
    crossings = 0

    for e in e2D:

        if crossing(p2D,p2D2,e):
            boolc = edge_cross(p2D,p2D2,e[:2],e[2:])
            if boolc:
                crossings+=1
    if np.mod(crossings,2)==0:
        return False
    else:
        return True


def sort_pts(pts):
    com = np.sum(pts,0)*0.25
    pt_s = np.zeros((4,np.shape(pts)[1]+1))
    pt_s[:,:2] = pts
    pt_s[:,2] = np.mod(np.arctan2(pt_s[:,1]-com[1],pt_s[:,0]-com[0])*180/np.pi,360)
    pt_s = sorted(pt_s,key=itemgetter(2))
    return np.array(pt_s)[:,:2]
    
if __name__ == "__main__":
    a,b = np.array([2*np.random.random(),2*np.random.random()]),2*np.array([np.random.random(),2*np.random.random()])
    pts_o = np.array([np.zeros(2),a,a+b,b])
    pts = sort_pts(pts_o)
    plane = np.array([0,0,1,0,0,0])
    pts2 = np.array([[np.random.random()*3-.5,np.random.random()*3-0.5,0.0] for ii in range(10)])
#    pts2 = np.array([[0.5,-1.5,0]])
    edges = np.array([[*pts[np.mod(ii,4)],*pts[np.mod(ii+1,4)]] for ii in range(4)])
    inside = []
    outside = []
    for p in pts2:
#        if in_pgram2D(p,np.array([0,0,1]),edges,5):
        if in_pgram(p,plane,edges,5):
            inside.append([p[0],p[1]])
        else:
            outside.append([p[0],p[1]])
    inside = np.array(inside)
    outside = np.array(outside)
#    pts = np.array([list(pts)+np.zeros(3)])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.scatter(pts[:,0],pts[:,1],c='b')
    ax.plot(edges[:,0],edges[:,1])
    ax.scatter(inside[:,0],inside[:,1],c='r')
    ax.scatter(outside[:,0],outside[:,1],c='k')
#    ax.axis([-0.5,1.5,-0.5,1.5])