# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 12:22:03 2018

@author: rday

Is a specified point in R3 inside a convex form?
To confirm, we find the number of times that a line emitted from the point
passes through one of the faces of the form. If mod(number,2)==0, then
we have a point outside. Else, if mod(number,2)==1, then we have found a point
inside the shape.

1. Minlen -- the longest line connecting 2 vertices of the shape. This defines
            the minimal length for our probe line.
2. Plane Cross -- Does the line cross any of the planes?

3. Location of Cross -- Find the solution of intersection of line with plane

4. Cross Inside -- Subroutine, performing similar test to see if the crossing 
                point is inside the bounded 2-dimensional plane which constitutes
                the face
5. Sum All Crossings -- Determine if an even number of crossings occur

6. Repeat -- for all points of interest to determine if they are contained by the 
            shape.


"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import parallelogram

class parallelepiped:
    
    def __init__(self,avecs):
        self.points,self.maxlen = self.calc_points(avecs)
        self.planes = self.calc_planes(avecs)
        self.edges,self.Rmat = self.calc_edges()
        
        
    def calc_points(self,avec):
        pts = np.array([[int(i/4),int(np.mod(i/2,2)),int(np.mod(i,2))] for i in range(8)])
        vert = np.dot(pts,avec)
        maxlen = 100*np.array([np.linalg.norm(vert[ii]-vert[jj]) for ii in range(len(vert)) for jj in range(ii+1,len(vert))]).max()
        return vert,maxlen
    
    def calc_planes(self,avec):
        planes = []
        for i in range(len(avec)):
            for j in range(i+1,len(avec)):
                tmp = np.cross(avec[i],avec[j])
                tmp = tmp/np.linalg.norm(tmp)
                if tmp[2]<0:
                    tmp = -1*tmp
                point = int([k for k in range(3) if (k!=i and k!=j)][0])
                planes.append([*tmp,*np.zeros(3),*avec[i],*(avec[i]+avec[j]),*avec[j]])
                planes.append([*tmp,*avec[point],*(avec[point]+avec[i]),*(avec[point]+avec[i]+avec[j]),*(avec[point]+avec[j])])
                
        return np.array(planes)
    
    
    def calc_edges(self):
        '''
        Edges are defined here by a beginning and end in R2. The coordinates are within the plane of the 
        plane itself, rather than 001 generically. This is a projected plane.
        For each plane, there should be 4 edges in a parallelepiped,
        so there will be 6 x 4 edges, each containing 2 len(2) vectors
        args:
            self (containing the array of planes)
        return:
            edges (numpy array shape 6x4x4 containing origin and end of edge)
        '''
        edges = np.zeros((len(self.planes),4,4))
        Rmatrices = np.zeros((len(self.planes),3,3))
        for p in range(len(self.planes)):
            Rmat = parallelogram.rotation(self.planes[p,:3])

            corners = np.dot(np.array([self.planes[p,3*(j+1):3*(j+2)] for j in range(4)]),Rmat)[:,:2]
            corners = parallelogram.sort_pts(corners)
            edges[p] = np.array([[*corners[np.mod(ii,4)],*corners[np.mod(ii+1,4)]] for ii in range(4)])
            Rmatrices[p] = Rmat
        return edges,Rmatrices
    

def sign_proj(point,plane):
    return np.sign(np.dot(point-plane[3:6],plane[:3]))

def cross_plane(p1,p2,plane):
    s1 = sign_proj(p1,plane)
    s2 = sign_proj(p2,plane)
    if s1!=s2:
        return True
    elif s1==s2 and s1!=0:
        return False
    elif s1==s2 and s1==0:
        print('Both in plane!')
        return False
    
    
def plane_intersect(p1,p2,plane):
    '''
    Find point of intersection of plane and line
    args:
        p1,p2 -- endpoints of the line segment in question (numpy array of 3 float)
        plane -- plane being intersected numpy array of 6 float (first 3 define planar orientation, second 3 are a point in plane)
    return:
        numpy array of 3 float indicating the point where the plane and line intersect
        
    '''
    norm = plane[:3]/np.linalg.norm(plane[:3])
    xo = plane[3:6]
    m = p2-p1
    b = p1
    return b-np.dot(norm,(-xo+b))/np.dot(norm,m)*m 




def inside_pped(pped,point):
    '''
    Check if a given point is in the defined parallelogram:
        1. Define a line-segment projecting from the point of interest.
        2. Iterating over each of the planes, does the line segment cross the plane?
        3. If so, does the crossing point occur within the plane of the parallelogram defining the plane?
        4. If so, the line segment crosses the plane.
        5. Sum over all instances of valid crossings so defined. If sum is even, outside the parallelepiped, else, inside.
    args: 
        pped -- instance of parallelepiped object
        point -- numpy array of 3 float corresponding to the point of interest
        
    return:
        Boolean True (inside) False (outside)
    '''
#    if np.linalg.norm(point)==0: 
#        return True
#    else:
    crossings = 0
    point2 = point + pped.maxlen*2*np.array([0,0,1])
    for pi in range(len(pped.planes)):
        if cross_plane(point,point2,pped.planes[pi]):
            intersect = plane_intersect(point,point2,pped.planes[pi])
            inter_2D = np.dot(intersect,pped.Rmat[pi])[:2]
            in_plane = parallelogram.in_pgram(inter_2D,pped.edges[pi],pped.maxlen)
            if in_plane:
                crossings+=1
    if np.mod(crossings,2)==0:
        return False
    else:
        return True


if __name__=="__main__":
    avecs = np.array([-0.5+1*np.random.random() for i in range(9)]).reshape((3,3))
##    avecs = np.array([[1,1,0],[1,-1,0],[0,0,1]])
#    avecs = av_bad
    pp = parallelepiped(avecs)
#    
#    points = np.array([[np.random.random()*4-0.5,np.random.random()*4-0.5,np.random.random()*4-0.5] for ii in range(2000)])
##    points = np.array([[1,2,2]])
    inside =[]
    outside = []
    for ii in range(1000):
        pi = np.array([np.random.random()*1-0.2,np.random.random()*1-0.2,np.random.random()*1-0.2])
        if inside_pped(pp,pi):
            inside.append(pi)
#        else:
#            outside.append(pi)
    inside = np.array(inside)
#    outside = np.array(outside)
    if len(inside)>0:
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        ax.scatter(pp.points[:,0],pp.points[:,1],pp.points[:,2],s=20,c='k')
    #    ax.scatter(points[:,0],points[:,1],points[:,2])
        ax.scatter(inside[:,0],inside[:,1],inside[:,2],c='r')
#        ax.scatter(outside[:,0],outside[:,1],outside[:,2],c='grey',alpha=0.3)