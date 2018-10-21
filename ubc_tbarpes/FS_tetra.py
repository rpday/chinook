# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 08:27:49 2018


@author: rday

FERMI SURFACE TRIANGULATION,
USING THE TETRAHEDRAL MESH APPROACH


"""



import ubc_tbarpes.tetrahedra as tetrahedra
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.tri as mtri
from operator import itemgetter



def EF_tetra(TB,NK,EF):
    '''
    Generate a tetrahedra mesh of k-points which span the BZ with even distribution
    Diagonalize over this mesh and then compute the resulting density of states as
    prescribed in the above paper. 
    The result is plotted, and DOS returned
    args:
        TB -- tight-binding model object
        NK -- integer / list of 3 integers -- number of k-points in mesh
    return:
    '''
    kpts,tetra = tetrahedra.mesh_tetra(TB.avec,NK,True)
    TB.Kobj.kpts = kpts
    TB.solve_H()
    surfaces = {bi:{'pts':[],'tris':[]} for bi in range(len(TB.basis)) if (TB.Eband[:,bi].min()<=EF and TB.Eband[:,bi].max()>=EF)} #iinitialize Fermi surface only for those bands which actually have one...
    for bi in surfaces.keys(): #iterate only over bands which have Fermi surface
        for ki in range(len(tetra)):
            E_tmp = TB.Eband[tetra[ki],bi]
            register = np.zeros((4,2))
            register[:,1] = tetra[ki]
            register[:,0] = E_tmp
            register = np.array(sorted(register,key=itemgetter(0)))
            kinds = register[:,1].astype(int)
            t = register[:,0]

            if t[0]<=EF and EF<=t[3]: #Fermi level is inside this tetrahedra
                if t[0]<=EF<t[1]:
                    k = kpts[kinds]
                    t5 = k[0]+(EF-t[0])/(t[2]-t[0])*(k[2]-k[0])
                    t6 = k[0] + (EF-t[0])/(t[3]-t[0])*(k[3]-k[0])
                    t7 = k[0] + (EF-t[0])/(t[1]-t[0])*(k[1]-k[0])
                
                    tri = [np.around([t5,t6,t7],4)]
                    
                elif t[1]<=EF<t[2]:
                    k = kpts[kinds]
                    t5 = k[0]+(EF-t[0])/(t[2]-t[0])*(k[2]-k[0])
                    t6 = k[0] + (EF-t[0])/(t[3]-t[0])*(k[3]-k[0])
                    t7 = k[1] +(EF-t[1])/(t[3]-t[1])*(k[3]-k[1])
                    t8 = k[1] + (EF-t[1])/(t[2]-t[1])*(k[2]-k[1])
                    tri = np.around(sim_tri([t5,t6,t7,t8]),4)
                
                elif t[2]<=EF<=t[3]:
                    k = kpts[kinds]
                    t5 = k[0] + (EF-t[0])/(t[3]-t[0])*(k[3]-k[0])
                    t6 = k[1] + (EF-t[1])/(t[3]-t[1])*(k[3]-k[1])
                    t7 = k[2] + (EF-t[2])/(t[3]-t[2])*(k[3]-k[2])
                    tri = [np.around([t5,t6,t7],4)]
                else:
                    continue
##                if cFS_ter(k[:,0],k[:,1],k[:,2])
#                    count+=1
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
                    
#                    ccoord = [surfaces[bi]['pts'][c] for c in coords]
#                    print(np.around(trii,4))
#                    print('\n')
#                    print(np.around(ccoord,4))
#                    print('------------------')

        surfaces[bi]['pts'] = np.array(surfaces[bi]['pts'])
        surfaces[bi]['tris'] = np.array(surfaces[bi]['tris'])

    
    return surfaces



def FS_generate(TB,Nk,EF):
    
    surfaces = EF_tetra(TB,Nk,EF)
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    for bi in surfaces:
        Fk=surfaces[bi]['pts']
        ax.plot_trisurf(Fk[:,0],Fk[:,1],Fk[:,2],triangles=surfaces[bi]['tris'],cmap=cm.magma,linewidth=0.5,edgecolor='white')
    

        
                
                                    






def heron(vert):
    L = [np.linalg.norm(vert[np.mod(j,3)]-vert[np.mod(j-1,3)]) for j in range(3)]
    S = 0.5*sum(L)
    return np.sqrt(S*(S-L[0])*(S-L[1])*(S-L[2]))


def sim_tri(vert):
    '''
    Take 4 vertices of a quadrilateral and split into two alternative triangulations of the corners.
    Return the vertices of the triangulation with the more similar areas
    args:
        vert -- list-like array of 4 points in some coordinate frame (float type)
    return
        tris[0],tris[1] -- two arrays of size 3x3 containing the coordinates of a triangulation float type
    '''
    
    tris = np.array([[vert[0],vert[2],vert[1]],[vert[0],vert[3],vert[2]],[vert[0],vert[3],vert[1]], [vert[1],vert[3],vert[2]]])
    areas = [heron(ti) for ti in tris]
    if abs(areas[1]-areas[0])<abs(areas[3]-areas[2]):
        return [tris[0],tris[1]]
    else:
        return [tris[2],tris[3]]
                
            
            

