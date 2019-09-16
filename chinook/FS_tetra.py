# -*- coding: utf-8 -*-
#Created on Sat Oct 20 08:27:49 2018

#Created on Tue Mar 26 20:42:09 2019

#@author: ryanday


#MIT License

#Copyright (c) 2018 Ryan Patrick Day

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



#FERMI SURFACE TRIANGULATION,
#USING THE TETRAHEDRAL MESH APPROACH, motivated by the discussion in Appendix A of
#Helmholtz Fermi Surface Harmonics: an efficient approach for treating anisotropic problems involving Fermi surface integrals
#May 2014New Journal of Physics
#DOI: 10.1088/1367-2630/16/6/063014




import chinook.tetrahedra as tetrahedra
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from operator import itemgetter
import warnings
warnings.filterwarnings("error")


    


def EF_tetra(TB,NK,EF,degen=False,origin=None):
    
    '''
    Generate a tetrahedra mesh of k-points which span the BZ with even distribution
    Diagonalize over this mesh and then compute the resulting density of states as
    prescribed in the above paper. 
    
    *args*:

        - **TB**: tight-binding model object
        
        - **NK**: int or list,tuple of 3 int indicating number of k-points in mesh
        
        - **EF**: float, Fermi energy, or energy of interest
        
    *kwargs*:

        - **degen**: bool, flag for whether the bands are two-fold degenerate, as for 
        Kramers degeneracy
        
        - **origin**: numpy array of 3 float,  corresponding to the desired centre of the 
        plotted Brillouin zone
        
    *return*:

        - **surfaces**: dictionary of dictionaries: Each key-value pair corresponds
        to a different band index. For each case, the value is a dictionary with key-value
        pairs:
            
            - *'pts'*: numpy array of Nx3 float, the N coordinates of EF crossing
            
            - *'tris'*: numpy array of Nx3 int, the triangulation of the surface
    
    ***
    '''
    
    kpts,tetra = tetrahedra.mesh_tetra(TB.avec,NK)
    if origin is not None:
        com = np.mean(kpts,axis=0)
        shift = origin-com
        kpts+=shift

    TB.Kobj.kpts = kpts
    TB.solve_H()
    if degen:
        inc = 2
        TB.Eband = TB.Eband[:,::inc]
        
    else:
        inc = 1
    surfaces = {bi:{'pts':[],'tris':[]} for bi in range(len(TB.basis[::inc])) if (TB.Eband[:,bi].min()<=EF and TB.Eband[:,bi].max()>=EF)} #iinitialize Fermi surface only for those bands which actually have one...

    for bi in surfaces.keys(): #iterate only over bands which have Fermi surface
        for ki in range(len(tetra)):
            E_tmp = TB.Eband[tetra[ki],bi] #vertex energies

            register = np.zeros((4,2))
            register[:,1] = tetra[ki]
            register[:,0] = E_tmp
            register = np.array(sorted(register,key=itemgetter(0))) #sort corners based on energies
            kinds = register[:,1].astype(int) #indices
            t = register[:,0] #energies

            if t[0]<=EF and EF<=t[3]: #if True, Fermi level cuts through this tetrahedra
                k = kpts[kinds] #get k-coordinates
                if t[0]<=EF<t[1]:
                    
                    t5 = k[0]+(EF-t[0])/(t[2]-t[0])*(k[2]-k[0])
                    t6 = k[0] + (EF-t[0])/(t[3]-t[0])*(k[3]-k[0])
                    t7 = k[0] + (EF-t[0])/(t[1]-t[0])*(k[1]-k[0])
                
                    tri = [np.around([t5,t6,t7],4)]
                    
                elif t[1]<=EF<t[2]:
                
                    t5 = k[0]+(EF-t[0])/(t[2]-t[0])*(k[2]-k[0])
                    t6 = k[0] + (EF-t[0])/(t[3]-t[0])*(k[3]-k[0])
                    t7 = k[1] +(EF-t[1])/(t[3]-t[1])*(k[3]-k[1])
                    t8 = k[1] + (EF-t[1])/(t[2]-t[1])*(k[2]-k[1])
                    tri = np.around(sim_tri([t5,t6,t7,t8]),4)
                
                elif t[2]<=EF<=t[3]:
                    
                    t5 = k[0] + (EF-t[0])/(t[3]-t[0])*(k[3]-k[0])
                    t6 = k[1] + (EF-t[1])/(t[3]-t[1])*(k[3]-k[1])
                    t7 = k[2] + (EF-t[2])/(t[3]-t[2])*(k[3]-k[2])
                    tri = [np.around([t5,t6,t7],4)]
                else:
                    continue
                for trii in tri:
                    coords = np.zeros(3,dtype=int)
                    for t in list(enumerate(trii)):
                        if len(surfaces[bi]['pts'])>0:
                            dv_bi = np.linalg.norm(t[1]-surfaces[bi]['pts'],axis=1) 
                        else: 
                            dv_bi=np.array([1])
                        if dv_bi.min()>1e-10:
                            surfaces[bi]['pts'].append(t[1])
                            coords[t[0]] = len(surfaces[bi]['pts'])-1
                        
                        
                        else:
                            coords[t[0]] =np.where(dv_bi==0)[0][0] 
                    surfaces[bi]['tris'].append(coords)
        surfaces[bi]['pts'] = np.array(surfaces[bi]['pts'])
        surfaces[bi]['tris'] = np.array(surfaces[bi]['tris'])

    
    return surfaces



def FS_generate(TB,Nk,EF,degen = False,origin=None,ax=None):
    
    '''

    Wrapper function for computing Fermi surface triangulation, and then plotting
    the result. 
    
    *args*:

        - **TB**: tight-binding model object
        
        - **Nk**: int, or tuple/list of 3 int, number of k-points in Brillouin zone mesh
        
        - **EF**: float, Fermi energy, or constant energy level of interest
        
    *kwargs*:

        - **degen**: bool, flag for whether the bands are two-fold degenerate, as for 
        Kramers degeneracy
        
        - **origin**: numpy array of 3 float,  corresponding to the desired centre of the 
        plotted Brillouin zone
        
        - **ax**: matplotlib Axes, option for plotting onto existing Axes

    
    *return*:

        - **surfaces**: dictionary of dictionaries: Each key-value pair corresponds
        to a different band index. For each case, the value is a dictionary with key-value
        pairs:
            
            - *'pts'*: numpy array of Nx3 float, the N coordinates of EF crossing
            
            - *'tris'*: numpy array of Nx3 int, the triangulation of the surface
            
        - **ax**: matplotlib Axes, for further modification

    ***
    '''
    
    surfaces = EF_tetra(TB,Nk,EF,degen,origin)
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
    for bi in surfaces:
        Fk=surfaces[bi]['pts']

        ax.plot_trisurf(Fk[:,0],Fk[:,1],Fk[:,2],cmap=cm.magma,triangles=surfaces[bi]['tris'],linewidth=0.1,edgecolor='w',alpha=1.0)
   # ax.grid(False)
#    ax.axis('off')
   # ax.set_aspect(1)
    return surfaces,ax


def heron(vert):
    '''
    Heron's algorithm for calculation of triangle area, defined by only the vertices
    
    *args*:

        - **vert**: numpy array of 3x3 indicating vertices of triangle
    
    *return*:

        - float, area of triangle
        
    ***
    '''
    
    L = [np.linalg.norm(vert[np.mod(j,3)]-vert[np.mod(j-1,3)]) for j in range(3)]
    S = 0.5*sum(L)
    
    try:
        area = np.sqrt(abs(S*(S-L[0])*(S-L[1])*(S-L[2])))
    except RuntimeWarning:
        print(S,L)
        area = 0
    return area


def sim_tri(vert):
    
    '''
    
    Take 4 vertices of a quadrilateral and split into two alternative triangulations of the corners.
    Return the vertices of the triangulation which has the more similar areas between the two
    triangles decomposed.
    
    *args*:

        - **vert**: (4 by 3 numpy array (or list) of float) in some coordinate frame 
        
    *return*:
    
        - **tris[0]** , **tris[1]**: two numpy arrays of size 3 by 3 float containing
        the coordinates of a triangulation 
    
    ***
    '''
    
    tris = np.array([[vert[0],vert[2],vert[1]],[vert[0],vert[3],vert[2]],[vert[0],vert[3],vert[1]], [vert[1],vert[3],vert[2]]])
    areas = [heron(ti) for ti in tris]
    if abs(areas[1]-areas[0])<abs(areas[3]-areas[2]):
        return [tris[0],tris[1]]
    else:
        return [tris[2],tris[3]]
    
    
#def FS_contour(TB,kmax,Ef):
#    kr = np.linspace(kmax/1000,kmax,100)
#    phi  = np.linspace(0,2*np.pi,100)
#    k_phi = np.zeros((len(phi),3))
#    for 
#    kvecs = np.array([[kr[i]*np.cos(phi[j]),kr[i]*np.sin(phi[j]),0] for j in range(len(phi)) for i in range(len(kr))])


#    TB.Kobj.kpts = kvecs
        
    
    