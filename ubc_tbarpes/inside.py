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
import ubc_tbarpes.parallelogram as parallelogram

class parallelepiped:
    
    def __init__(self,avecs):
        self.avecs = avecs
        self.points,self.maxlen = self.calc_points(avecs)
        self.planes = self.calc_planes(avecs)
        self.edges,self.Rmat = self.calc_edges()
        self.bounding = self.define_lines()
        
        
    def calc_points(self,avec):
        pts = np.array([[int(i/4),int(np.mod(i/2,2)),int(np.mod(i,2))] for i in range(8)])
        vert = np.dot(pts,avec)
        maxlen = 1e6*np.array([np.linalg.norm(vert[ii]-vert[jj]) for ii in range(len(vert)) for jj in range(ii+1,len(vert))]).max()
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
    
    def define_lines(self):
        lines = np.zeros((24,2,3))
        for ii in range(24):
            pts = self.planes[int(ii/4)][3:]
            j = np.mod(ii,4)
            in1 = np.mod((j)*3,12),np.mod(3*(j+1),12), np.mod((j+1)*3,12),np.mod(3*(j+2),12)
            inds = [jj if jj>0 else None for jj in in1]
            tmp=np.array([pts[inds[0]:inds[1]],pts[inds[2]:inds[3]]])
            lines[ii] = tmp
        return lines
    
    


    
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
            Rmat = rotation(self.planes[p,:3])

            corners = np.dot(np.array([self.planes[p,3*(j+1):3*(j+2)] for j in range(4)]),Rmat)[:,:2]
            corners = parallelogram.sort_pts(corners)
            edges[p] = np.array([[*corners[np.mod(ii,4)],*corners[np.mod(ii+1,4)]] for ii in range(4)])
            Rmatrices[p] = Rmat
        return edges,Rmatrices
    
    
def _draw_lines(ax,pp):
    for i in range(24):
        ax.plot(pp.bounding[i][:,0],pp.bounding[i][:,1],pp.bounding[i][:,2],c='k')
    
    
def rotation(norm):
    '''
    Rotate the plane so that it is normal to the 001 direction, then strip
    the z-information.
    '''
    norm = norm/np.linalg.norm(norm)
    
    if abs(norm[2])!=np.linalg.norm(norm):
        x = np.cross(norm,np.array([0,0,1]))
        sin = np.linalg.norm(x)
        x = x/sin
        cos = np.dot(norm,np.array([0,0,1]))
    else:
        x = np.array([1,0,0])
        if norm[2]>0:
            
            cos,sin=1,0
        elif norm[2]<0:
            cos,sin=-1,0
    u = np.array([[0,-x[2],x[1]],[x[2],0,-x[0]],[-x[1],x[0],0]])
    uu = np.array([x[0]*x,x[1]*x,x[2]*x])
    R = cos*np.identity(3) + sin*u + (1-cos)*uu

    return R.T

    

def sign_proj(point,plane):
    '''
    Define a 'side' of the plane which point is on. To do so, simply take dot product of the
    point (referenced to a point in the plane) with the normal vector of the plane.
    args:
        point -- numpy array of 3 float
        plane -- plane attribute for the parallelepiped, as defined in the class above. (numpy array of 15 float)
    return:
        sign (+/-1,0) corresponding to above, in or below the plane
    '''
    return np.sign(np.around(np.dot(point-plane[3:6],plane[:3]),3))

def cross_plane(p1,p2,plane):
    '''
    Does a line segment connecting p1 to p2 cross a plane? To do so, check the sign of the projection of the 
    two points on the plane, and look for a change along the linesegment connecting these points.
    args:
        p1,p2 -- numpy array of 3 float
        plane -- plane attribute for the parallelepiped, as defined in the class above. (numpy array of 15 float)
    return:
        bool True if line crosses plane, else False
    '''
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
    return p1-np.dot(norm,(p1-xo))/np.dot(norm,m)*m 

def point_is_intersect(point,intersect):
    if np.linalg.norm(point-intersect)==0.0:
        return True
    else:
        return False

def origin_plane(pi):
    if np.mod(pi,2)==0:
        return 1
    else:
        return 0
        


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
#    if sum(True if (np.linalg.norm(point-pped.avecs[ii]*np.dot(point,pped.avecs[ii])/np.linalg.norm(pped.avecs[ii])**2)==0.0 and np.linalg.norm(point)<np.linalg.norm(pped.avecs[ii])) else False for ii in range(3)):
#        return True
#    else:
    point = np.around(point,4)
    crossings = 0
    point2 = point + pped.maxlen*2*np.array([149,151,157])
    for pi in range(len(pped.planes)):
            
        if cross_plane(point,point2,pped.planes[pi]):
            intersect = plane_intersect(point,point2,pped.planes[pi])
            inter_2D = np.dot(intersect,pped.Rmat[pi])[:2]
            in_plane = parallelogram.in_pgram(inter_2D,pped.edges[pi],pped.maxlen)

            if in_plane:
               if point_is_intersect(point,intersect):#IF THE POINT IS CONTAINED IN A PLANE
                    crossings = origin_plane(pi)
                    break
               crossings+=1
    if np.mod(crossings,2)==0:
        return False
    else:
        return True


if __name__=="__main__":
    print('run')
#    avecs = np.array([[  4.101698,  -2.368118,   0.      ],
#       [  0.      , -18.94494 ,  13.492372],
#       [ -4.101698,  -2.368117, -13.492372]])
#
#    pp = parallelepiped(avecs)
#    pts = np.zeros((3,10))
#    inside =[]
#    outside = []
#    for ii in range(len(pts)):
#        pi = np.array([np.random.random()*10-5,np.random.random()*30-27,np.random.random()*30-15])
#        if inside_pped(pp,pi):
#            inside.append(pi)
#        else:
#            outside.append(pi)
#    inside = np.array(inside)
#    outside = np.array(outside)
#    if len(inside)>0:
#        fig = plt.figure()
#        ax = fig.add_subplot(111,projection='3d')
#        ax.scatter(pp.points[:,0],pp.points[:,1],pp.points[:,2],s=20,c='k')
#    #    ax.scatter(points[:,0],points[:,1],points[:,2])
#        ax.scatter(inside[:,0],inside[:,1],inside[:,2],c='r')
#        ax.scatter(outside[:,0],outside[:,1],outside[:,2],c='grey',alpha=0.3)