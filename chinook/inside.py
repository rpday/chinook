# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 12:22:03 2018
@author: rday
Is a specified point in R3 inside a convex form?
To confirm, we find the number of times that a line emitted from the point
passes through one of the faces of the form. If mod(number,2)==0, then
we have a point outside. Else, if mod(number,2)==1, then we have found a point
inside the shape. In this script, we solve this problem for the case of a parallelepiped
formed by 3 spanning vectors in R3

"""


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import chinook.parallelogram as parallelogram
import chinook.rotation_lib as rot_lib

class parallelepiped:
    
    '''
    The parallelepiped object defines the bounding region formed by three spanning
    vectors in R3, defined as a numpy array of 3x3 float, **avecs**. The methods
    here establish the boundary edges and planes
    
    '''
    
    def __init__(self,avecs):
        
        '''
        Initialize the parallelepiped object
        
        *args*:
            - **avecs**: numpy array of 3x3 float, spanning vectors
            
        ***
        '''
        self.avecs = avecs
        self.points,self.maxlen = self.calc_points()
        self.planes = self.calc_planes()
        self.edges,self.Rmat,self.edge_zeros = self.calc_edges()
        self.bounding = self.define_lines()
        
        
    def calc_points(self):
        '''
        Find the 8 corners of the parallelpiped, and define a definitively 
        excessive path length for use in defining boundary crossings
        
        *return*:
            - **vert**: numpy array 8x3 float, vertices of form
            
            - **maxlen**: float, max length of a line segment used in finding points
            inside
            
        ***
        '''
        pts = np.array([[int(i/4),int(np.mod(i/2,2)),int(np.mod(i,2))] for i in range(8)])
        vert = np.dot(pts,self.avecs)
        maxlen = 1e3*np.array([np.linalg.norm(vert[ii]-vert[jj]) for ii in range(len(vert)) for jj in range(ii+1,len(vert))]).max()
        return vert,maxlen
    
    def calc_planes(self):
        '''
        Define the planes. They are stored as an array of 15 float: 
        [0:3] normal, [3:6],[6:9],[9:12],[12:] are four corners of each plane
        
        *return*:
            - **planes**: numpy array of 6x15 float carrying norm and coordinates of corners of the plane
        
        *** 
        '''
        planes = []
        for i in range(len(self.avecs)):
            for j in range(i+1,len(self.avecs)):
                cross_prod = np.cross(self.avecs[i],self.avecs[j])
                cross_prod = cross_prod/np.linalg.norm(cross_prod)
                if cross_prod[2]<0:
                    cross_prod = -1*cross_prod
                point = int([k for k in range(3) if (k!=i and k!=j)][0])
                planes.append([*cross_prod,*np.zeros(3),*self.avecs[i],*(self.avecs[i]+self.avecs[j]),*self.avecs[j]])
                planes.append([*cross_prod,*self.avecs[point],*(self.avecs[point]+self.avecs[i]),*(self.avecs[point]+self.avecs[i]+self.avecs[j]),*(self.avecs[point]+self.avecs[j])])
                
        return np.array(planes)
    
    def define_lines(self):
        '''
        For plotting purposes only, define the line segments which form 
        the boundary of the parallelepiped.
        
        *return*:
            - **lines**: numpy array of float, size 24x2x3 indicating
            the 2 endpoints in R3 of each of the 24 lines which constitute
            a parallelepiped
            
        ***
        ''' 
        
        lines = np.zeros((24,2,3))
        for ii in range(24):
            pts = self.planes[int(ii/4)][3:]
            ii_mod4 = np.mod(ii,4)
            in1 = np.mod((ii_mod4)*3,12),np.mod(3*(ii_mod4+1),12), np.mod((ii_mod4+1)*3,12),np.mod(3*(ii_mod4+2),12)
            inds = [jj if jj>0 else None for jj in in1]
            endpoints = np.array([pts[inds[0]:inds[1]],pts[inds[2]:inds[3]]])
            lines[ii] = endpoints
        return lines
    
    


    
    def calc_edges(self):
        '''
        Edges are defined here by a beginning and end in R2. The coordinates 
        are within the plane of the plane itself, rather than 001.
        This is a projected plane. For each plane, there are 4 edges so there
        will be 6 x 4 edges
        
        *return*:
            - **edges**: numpy array of float, shape 6x4x4 containing origin
            and end of edge), each of the len 4 arrays are the two endpoints
            R2
            
            - **Rmatrices**: numpy array of 3x3 float, rotation matrices which 
            bring the plane into the normal 001 direction
            
            - **edge_origins**: numpy array of integers (6x2) indicating which 2 edges
            of a given plane should be taken as the origin edges (i.e. a point which
            touches one of these edges is considered to be 'inside', but for the other
            two, such is not the case). This is only relevant for planes which include
            a lattice vector, and so for other faces, **edge_origins** entry will
            be (-1,-1) since no edges are valid.
            
        ***
        '''
        edges = np.zeros((len(self.planes),4,4))
        Rmatrices = np.zeros((len(self.planes),3,3))
        edge_origins = np.zeros((len(self.planes),2))
        for p in range(len(self.planes)):
            Rmat = rot_lib.rotate_v1v2(np.array([0,0,1]),self.planes[p,:3])
#            Rmat = rotation(self.planes[p,:3])

            corners = np.dot(np.array([self.planes[p,3*(j+1):3*(j+2)] for j in range(4)]),Rmat)[:,:2]
            corners = parallelogram.sort_pts(corners)
            edges[p] = np.array([[*corners[np.mod(ii,4)],*corners[np.mod(ii+1,4)]] for ii in range(4)])
            Rmatrices[p] = Rmat
            try:
                origin_ind = np.where(np.linalg.norm(corners,axis=1)==0)[0][0]
                edge_origins[p] = np.array([origin_ind,np.mod(origin_ind-1,4)])
            except IndexError:
                edge_origins[p] = np.array([-1,-1])
            
        return edges,Rmatrices,edge_origins
    
    
def _draw_lines(ax,pp):
    '''
    Utility script for plotting the bounding lines of the parallelepiped for 
    visualization purposes.
    
    *args*:
        - **ax**: the *matplotlib* axis on which the lines are to be plotted
        
        - **pp**: parallelepiped object, from which we take the bounding lines
    
    ***
    '''
    
    for i in range(24):
        ax.plot(pp.bounding[i][:,0],pp.bounding[i][:,1],pp.bounding[i][:,2],c='k')
    
def sign_proj(point,plane):
    
    '''
    Define a 'side' of the plane which point is on. To do so, simply take dot product of the
    point (referenced to a point in the plane) with the normal vector of the plane.
    
    *args*:
        - **point**: numpy array of 3 float
        
        - **plane**: numpy array of 15 float, plane attribute for the parallelepiped, 
        as defined in the class above. 
        
    *return*:
        - int (+/-1,0) corresponding to above, in or below the plane
        
    ***
    '''
    return np.sign(np.around(np.dot(point-plane[3:6],plane[:3]),3))

def cross_plane(p1,p2,plane):
    
    '''
    Does a line segment connecting p1 to p2 cross a plane? To do so, check the sign of the projection of the 
    two points on the plane, and look for a change along the linesegment connecting these points.
    
    *args*:
        - **p1**, **p2**: numpy array of 3 float
        
        - **plane**: numpy array of 15 float, **plane** attribute for the parallelepiped,
        as defined in the class above.
        
    *return*:
        - bool True if line crosses plane, else False
         
    ***
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
    
    *args*:
        - **p1**, **p2**: numpy array of 3 float, endpoints of the line segment
        in question 
        
        - **plane**: **plane** being intersected
        
    *return*:
        - numpy array of 3 float indicating the point where the plane and line 
        intersect in R3
         
    ***
    '''
    norm = plane[:3]/np.linalg.norm(plane[:3])
    xo = plane[3:6]
    m = p2-p1
    return np.around(p1-np.dot(norm,(p1-xo))/np.dot(norm,m)*m ,4)

def point_is_intersect(point,intersect):
    
    '''
    Is the point where the line segment intersects an edge equivalent to
    the point of interest?
    
    *args*:
        - **point**: numpy array of 3 float
        
        - **intersect**: numpy array of 3 float
        
    *return*:
        - bool, True if the two are equivalent to within desired numerical
        accuracy.
    
    ***    
    '''
    
    if np.linalg.norm(point-intersect)<1e-4:
        return True
    else:
        return False

def origin_plane(pi):
    
    '''
    The planes which include the origin are even numbered
    in index, by construction. If even, we have an origin plane. To be used in
    deciding whether a point contained in a plane should be taken as inside.
    
    *args*:
        - **pi**: int, plane index
        
    *return*:
        - int, 1 if pi even, 0 else
        
    ***
    '''
    
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
    
    *args*: 
        - **pped**: instance of parallelepiped object
        
        - **point**: numpy array of 3 float corresponding to the point of interest
        
    *return*:
        - bool True (inside) False (outside)
    
    ***
    '''

    point2 = np.around(point + pped.maxlen*2*np.array([149,151,157]),4) #linesegment is in an arbitrary direction, guaranteed to be outside the parallelepiped
    point = np.around(point,4)
    crossings = 0
    
    avi = np.linalg.inv(pped.avecs)
    for pi in range(len(pped.planes)):
        if is_lattice(point,avi):
            return False
        else:
            if cross_plane(point,point2,pped.planes[pi]):
                intersect = plane_intersect(point,point2,pped.planes[pi])
                inter_2D = np.dot(intersect,pped.Rmat[pi])[:2]
                in_plane = parallelogram.in_pgram(inter_2D,pped.edges[pi],pped.edge_zeros[pi],pped.maxlen)
                if in_plane<0:
                    crossings=0
                    break
                if in_plane>0:
                    if point_is_intersect(point,intersect):#IF THE POINT IS CONTAINED IN A PLANE
                        crossings = origin_plane(pi)
                        break
                    crossings+=1
    if np.mod(crossings,2)==0:
        return False
    else:
        return True
    

def is_lattice(point,avec_i):
    '''
    A quick check to see if a point is in fact a lattice point. 
    Then it should be excluded from consideration--except the origin
    
    *args*:
        - **p**: numpy array of 3 float, point of interest
        
        - **ai**: numpy array of 3x3 float, inverse lattice vector array,
        used to project the point **p** onto the lattice vectors
        
    *return*:
        - bool if lattice point, return True, else False
    
    ***  
    '''
    lattice_projection = np.around(np.dot(point,avec_i),4)
    if np.linalg.norm(lattice_projection.astype(int)-lattice_projection)==0 and np.linalg.norm(point)>0:
        return True
    else:
        return False
    
    
    
#    
#def rotation(norm):
#    '''
#    Rotate the plane so that it is normal to the 001 direction, then strip
#    the z-information.
#    '''
#    norm = norm/np.linalg.norm(norm)
#    
#    if abs(norm[2])!=np.linalg.norm(norm):
#        x = np.cross(norm,np.array([0,0,1]))
#        sin = np.linalg.norm(x)
#        x = x/sin
#        cos = np.dot(norm,np.array([0,0,1]))
#    else:
#        x = np.array([1,0,0])
#        if norm[2]>0:
#            
#            cos,sin=1,0
#        elif norm[2]<0:
#            cos,sin=-1,0
#    u = np.array([[0,-x[2],x[1]],[x[2],0,-x[0]],[-x[1],x[0],0]])
#    uu = np.array([x[0]*x,x[1]*x,x[2]*x])
#    R = cos*np.identity(3) + sin*u + (1-cos)*uu
#
#    return R.T


#if __name__=="__main__":
#    print('run')
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