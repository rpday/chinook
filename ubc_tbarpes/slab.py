# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 11:21:25 2018

@author: rday

MIT License

Copyright (c) 2018 Ryan Patrick Day

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""



import numpy as np
from operator import itemgetter
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import ubc_tbarpes.inside as inside
import ubc_tbarpes.orbital as olib
import ubc_tbarpes.TB_lib as TB_lib
import ubc_tbarpes.v3find as v3


def GCD(a,b):
    '''
    Basic greatest common denominator function. First find all divisors of each a and b.
    Then find the maximal common element of their divisors.
    args: a,b int
    return: GCD of a,b 
    '''
    dA,dB = divisors(a),divisors(b)
    return max(dA[i] for i in range(len(dA)) if dA[i] in dB)
    
def LCM(a,b):
    '''
    Basic lowest-common multiplier for two values a,b. Based on idea that LCM is just the
    product of the two input, divided by their greatest common denominator.
    args: a,b, int
    return: LCM
    '''
    return int(a*b/GCD(a,b))

def divisors(a):
    '''
    Iterate through all integer divisors of integer input
    args: a int
    return list of divisors
    '''
    return [int(a/i) for i in range(1,a+1) if (a/i)%1==0]




def LCM_3(a,b,c):
    '''
    For generating spanning vectors, require lowest common multiple of 3 integers, itself just
    the LCM of one of the numbers, and the LCM of the other two. 
    args: a,b,c int
    return LCM of the three numbers
    '''
    
    return LCM(a,LCM(b,c))

def iszero(a):
    
    return np.where(a==0)[0]

def nonzero(a):
    
    return np.where(a!=0)[0]

def abs_to_frac(avec,vec):
    '''
    Quick function for taking a row-ordered matrix of lattice vectors: 
        | a_11  a_12  a_13  |
        | a_21  a_22  a_23  |
        | a_31  a_32  a_33  |
    and using to transform a vector, written in absolute units, to fractional units.
    Note this function can be used to broadcast over N vectors you would like to transform
    args: avec -- numpy array of lattice vectors, ordered by rows (numpy array 3x3 of float)
          vec -- numpy array of 3 float (or Nx3) -- basis vector to be transformed to fractional coordinates
     return: re-defined basis vector (len 3 array of float), or Nx3 array of float
    
    '''
    return np.dot(vec,np.linalg.inv(avec))

def frac_to_abs(avec,vec):
    '''
    Same as abs_to_frac, but in opposite direction--from fractional to absolute coordinates
    args:
        avec -- numpy array of lattice vectors, row-ordered (3x3 numpy array of float)
        vec -- numpy array of input vector(s) as Nx3 float
    return: vec in units of absolute coordinates (Angstrom), N x 3 array of float
        
    '''
    return np.dot(vec,avec)


def p_vecs(miller,avec):
    '''
    Produce the vectors p, as defined by Ceder, to be used in defining spanning vectors for plane normal
    to the Miller axis
    args: miller numpy array of len 3 float
        avec numpy array of size 3x3 of float
    return p, numpy array size 3x3 of float
    '''
    n_zero = nonzero(miller)
    zero = iszero(miller)
    p = np.zeros((3,3))
    if len(n_zero)==3:
        M = LCM_3(*miller)
        p = np.array([avec[0]*M/miller[0],
                      avec[1]*M/miller[1],
                      avec[2]*M/miller[2]])
    elif len(n_zero)==2:
        M = LCM(*miller[n_zero])
        p[n_zero[0]] = M/miller[n_zero[0]]*avec[n_zero[0]]
        p[n_zero[1]] = M/miller[n_zero[1]]*avec[n_zero[1]]
        p[zero[0]] = p[n_zero[0]] + avec[zero[0]]
    elif len(n_zero)==1:
        p[n_zero[0]] = np.zeros(3)
        p[zero[0]] = avec[zero[0]]
        p[zero[1]] = avec[zero[1]]
        
    return p


def v_vecs(miller,avec):
    '''
    Wrapper for functions used to determine the vectors used to define the new,
    surface unit cell.
    args: 
        miller -- numpy array of int len(3) Miller indices for surface normal
        avec -- numpy array of 3x3 float -- Lattice vectors
    return:
        new surface unit cell vectors numpy array of 3x3 float
    '''
    p = p_vecs(miller,avec)
    v = np.zeros((3,3))
    v[0] = p[1]-p[0]
    v[1] = p[2]-p[0]
    v[2] = v3.find_v3(v[0],v[1],avec,40)
    return v

def basal_plane(v):
    '''
    Everything is most convenient if we redefine the basal plane of the surface normal to be oriented within a 
    Cartesian plane. To do so, we take the v-vectors. We get the norm of v1,v2 and then find the cross product with
    the z-axis, as well as the angle between these two vectors. We can then rotate the surface normal onto the z-axis.
    In this way we conveniently re-orient the v1,v2 axes into the Cartesian x,y plane.
    ARGS:
        v -- numpy array 3x3 float
    return:
        vp -- rotated v vectors, numpy array 3x3 of float
        R -- rotation matrix to send original coordinate frame into the rotated coordinates.
    '''
    norm = np.cross(v[0],v[1])
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
    R = R.T
    vp = np.dot(v,R)
    
    # Now perform one more rotation, taking vp[0] onto the [100] Cartesian axis
    phi = np.arccos(vp[0,0]/np.linalg.norm(vp[0]))
    Rp = np.array([[np.cos(phi),-np.sin(phi),0],[np.sin(phi),np.cos(phi),0],[0,0,1]]).T
    vp = np.dot(vp,Rp)
    R = np.dot(R,Rp)
    vp = np.around(vp,4)
    R = np.around(R,15)
    return vp,R


def par(avec):
    '''
    Definition of the parallelepiped, as well as a containing region within the 
    Cartesian projection of this form which can then be used to guarantee correct
    definition of the new cell basis. The parallelipiped is generated, and then
    its extremal coordinates established, from which a containing parallelepiped is
    then defined. The parallelepiped (blue) is plotted along with a semi-transparent
    (red) container box
    args:
        avec -- numpy array of 3x3 float
    return:
        vert -- numpy array  8x3 float vertices of parallelepiped
        box_pts -- numpy array 8 x 3 float vertices of containing box
    '''
    pts = np.array([[int(i/4),int(np.mod(i/2,2)),int(np.mod(i,2))] for i in range(8)])
    vert = np.dot(pts,avec)
    alpha,omega = np.array([vert[:,0].min(),vert[:,1].min(),vert[:,2].min()]),np.array([vert[:,0].max(),vert[:,1].max(),vert[:,2].max()])
    box = np.identity(3)*(omega-alpha)
    box_pts = np.dot(pts,box)
    box_pts = np.array([c+alpha for c in box_pts])
    return vert,box_pts

def populate_box(box,basis,avec,R):
    '''
    Populate the bounding box with points from the original lattice basis.
    
    '''
    box_av = np.dot(box,np.linalg.inv(avec))
    boxlims = np.array([[np.sign(box_av[:,i].min())*int(np.round(abs(box_av[:,i].min()))),np.sign(box_av[:,i].max())*int(np.round(abs(box_av[:,i].max())))] for i in range(3)])
    boxmesh = np.array([[boxlims[0,0]+i,boxlims[1,0]+j,boxlims[2,0]+k] 
                        for i in range(int(boxlims[0,1]-boxlims[0,0]+1))
                        for j in range(int(boxlims[1,1]-boxlims[1,0]+1))
                        for k in range(int(boxlims[2,1] - boxlims[2,0]+1))])
    real_space = np.dot(boxmesh,avec)
    
    basis_fill = []
    for ri in real_space:
        for b in basis:
            tmp = np.dot(b.pos,R) + ri
            basis_fill.append([*tmp,b.index])
    return np.array(basis_fill)

def populate_par(points,avec):
    '''
    Fill the box with basis points
    args:
        points - find the basis points, defined within the bounding box of the parallelepiped
                which are themselves inside the parallelepiped -- numpy array of Nx4 float (3 - position, 1 - index)
        avec - numpy array of float (3x3 float)
    return:
        new points, (Nx3 numpy array of float), indices (Nx1 numpy array of float)
        
    '''
    new_points = []
    indices = []
    pped = inside.parallelepiped(avec)
    for p in points:
        if inside.inside_pped(pped,p[:3]):
            new_points.append(p[:3])
            indices.append(p[-1])
    return np.array(new_points),np.array(indices)
        

def gen_cov_mat(avec,vvec):
    '''
    Generate the covariant transformation, as with cov_transform, not in use for ubc_tbarpes
    args:
        avec -- original lattice vectors (3x3 numpy array of float)
        vvec -- surface unit cell lattice vectors (3x3 numpy array of float)
    return:
        C_44 the covariant transformation matrix. The first 3x3 are the transformation of coordinates,
        the last row and column are empty except C[-1,-1] which maintains the index of the basis element, to 
        preserve its definition.
    '''
    C_33 = vvec*np.linalg.inv(avec)
    C_44 = np.zeros((4,4))
    C_44[:3,:3] = C_33
    C_44[3,3] = 1
    
    return C_44

def cov_transform(basis,indices,avec,vvec):
    '''
    Transform definition of states in terms of fractional coordinates into those for the redefined unit cell
    This is not in use, since we don't use fractional coordinates really. All in for the direct coordinates.
    '''
    Cmat = gen_cov_mat(avec,vvec)
    coords = np.zeros((len(indices),4))
    
    coords[:,:3] = np.array([basis[int(indices[ii])].fpos for ii in range(len(indices))])
    coords[:,-1] = indices
    new_coords = np.dot(Cmat,coords.T).T
    return new_coords[:,:3],new_coords[:,-1]


def gen_surface(avec,miller,basis):
    '''
    Construct the surface unit cell, to then be propagated along the 001 direction to form a slab
    args:
        avec - lattice vectors for original unit cell
        miller - Miller indices indicating the surface orientation
        basis - orbital basis for the original lattice
    return:
        new_basis - surface unit cell orbital basis
        vn_b - the surface unit cell primitive lattice vectors
        R -- rotation matrix, to be used in post-multiplication order numpy array 3x3 of float
    '''
    vn_b,R = basal_plane(v_vecs(miller,avec))
    pipe,box = par(vn_b)
    avec_R = np.dot(avec,R)
    uv,gamma = rot_vector(R.T)
    
    b_points = populate_box(box,basis,avec_R,R)
    in_pped,inds = populate_par(b_points,vn_b)
    new_basis = np.empty(len(in_pped),dtype=olib.orbital)
    ordering = sorted_basis(in_pped,inds)
    for ii in range(len(in_pped)):
        tmp = basis[int(ordering[ii][3])].copy()
        tmp.slab_index = int(ordering[ii][3])
        tmp.index = ii
        tmp.pos = ordering[ii][:3]
        tmp.proj,tmp.Dmat = tmp.rot_projection(-gamma,uv)
        new_basis[ii] = tmp
    
    
    return new_basis,vn_b,R

def rot_vector(Rmat):
    L,u=np.linalg.eig(Rmat)
    uv = np.real(u[:,np.where(abs(L-1)<1e-10)[0][0]])
    th = np.arccos((np.trace(Rmat)-1)/2)
    R_tmp = olib.Rmat(uv,th)
    if np.linalg.norm(R_tmp-Rmat)<1e-10:
        return uv,th
    else:
        R_tmp = olib.Rmat(uv,-th)
        if np.linalg.norm(R_tmp-Rmat)<1e-10:
            return uv,-th
        else:
            print('ERROR: COULD NOT DEFINE ROTATION MATRIX FOR SUGGESTED BASIS TRANSFORMATION!')
            return None
    
def sorted_basis(pts,inds):
    labels = np.array([[*pts[ii],inds[ii]] for ii in range(len(inds))])
    labels_sorted = np.array(sorted(labels,key=itemgetter(2,3)))
    return labels_sorted


def gen_slab(basis,vn,mint,minb,term,fine=(0,0)):
    '''
    Using the new basis defined for the surface unit cell, generate a slab
    of at least mint (minimum thickness), minb (minimum buffer) and terminated
    by orbital term. In principal the termination should be same on both top and
    bottom to avoid inversion symmetry breaking between the two lattice terminations.
    In certain cases, mint,minb may need to be tuned somewhat to get exactly the structure
    you want.
    args:
        basis - list of instances of orbital objects 
        vn - surface unit cell lattice vectors numpy array of 3x3 float
        mint - minimum thickness of the slab float
        minb - minimum thickness of the vacuum buffer float
        term - termination of the slab tuple (term[0] = top termination, term[1] = bottom termination)
        fine - fine adjustment of the termination to precisely specify terminating atoms: a tuple of 2 float indicating shift from the otherwise defined slab relative to its limits
    return:
        avec - updated lattice vector for the SLAB unit cell, numpy array of float (3x3)
        cull - array of new orbital basis objects, with slab-index corresponding to the original basis indexing
    '''
    pts = []
    Nmin = int(np.ceil((mint+minb)/vn[2,2])+2)
    
    for i in range(Nmin):
        for bi in basis:
#            pts.append([*(i*vn[-1]+bi.pos),bi.slab_index])
            pts.append([*(i*vn[-1]+bi.pos),bi.atom,bi.slab_index,bi.index])
    pts = np.array(pts)
    z_order = pts[:,2].argsort()
    pts = pts[z_order]
    
    pts = np.array(sorted(pts,key=itemgetter(2,5)))
    #termination organized by ATOM
    term_1set = pts[pts[:,3]==term[1]]
    term_0set = pts[pts[:,3]==term[0]]
    surf = pts[:,2].max()-minb

#    base = term_1set[:,2].min()
    base = term_1set[term_1set[:,2]>=(term_1set[:,2].min()+fine[1]),2].min()
#    top = term_0set[np.where(abs(term_0set[:,2]-surf)==abs(term_0set[:,2]-surf).min())[0][0],2]
    top = term_0set[np.where(abs(term_0set[(term_0set[:,2]-surf)<fine[0],2]-surf)==abs(term_0set[(term_0set[:,2]-surf)<fine[0],2]-surf).min())[0][0],2]

    cull = np.array([pts[p] for p in range(len(pts)) if base<=pts[p,2]<=top])
    cull[:,2]-=top
    avec = np.copy(vn)
    avec[2] = Nmin*vn[2]
    
    new_basis = np.empty(len(cull),dtype=olib.orbital)
    for ii in range(len(cull)):
        tmp = basis[int(cull[ii,-1])].copy()
        tmp.slab_index = int(cull[ii,-1])
        tmp.index = ii
        tmp.pos = cull[ii,:3]-np.array([cull[-1,0],cull[-1,1],0])
        tmp.depth = tmp.pos[2]
        new_basis[ii] = tmp
        
    return avec,new_basis


def region(num):
    '''
    Generate a symmetric grid of points in number of lattice vectors. The tacit assumption is a 3 dimensional lattice
    args: num -- integer--grid will have size 2*num+1 in each direction
    returns numpy array of size ((2*num+1)**3,3) with centre value of first entry of (-num,-num,-num),...,(0,0,0),...,(num,num,num)
    '''
    num_symm = 2*num+1
    return np.array([[int(i/num_symm**2)-num,int(i/num_symm)%num_symm-num,i%num_symm-num] for i in range((num_symm)**3)])


def unpack(H):
    '''
    Reduce a Hamiltonian object down to a list of matrix elements. Include the Hermitian conjugate terms
    args:
        H -- Hamiltonian object
    return
        Hlist -- list of Hamiltonian matrix elements
    '''
    Hlist =[]
    for hij in H:
        for el in hij.H:
            Hlist.append([hij.i,hij.j,*el])
    return Hlist
#
#
def H_surf(surf_basis,avec,H_bulk,Rmat):
    '''
    Rewrite the bulk-Hamiltonian in terms of the surface unit cell, with its (most likely expanded) basis.
    The idea here is to organize all 'duplicate' orbitals, in terms of their various connecting vectors.
    Using modular arithmetic, we then create an organized dictionary which categorizes the hopping paths
    within the new unit cell according to the new basis index designation. For each element in the Hamiltonian
    then, we can do the same modular definition of the hopping vector, easily determining which orbital in our
    new basis this hopping path indeed corresponds to. We then make a new list, organizing corresponding to the
    new basis listing.
    '''
    av_i = np.linalg.inv(avec)
    cv_dict = mod_dict(surf_basis,av_i)

    H_old = unpack(H_bulk)

    Rcv = np.dot(np.array([h[2:5] for h in H_old]),Rmat)
    H_new = []
    for ii in range(len(H_old)):
        hi = H_old[ii]
        R_latt = np.mod(np.around(np.dot(Rcv[ii],av_i),4),1)
#        R_latt = np.around(np.dot(Rcv[ii],av_i),4)
        R_compare = (np.linalg.norm((cv_dict['{:d}-{:d}'.format(hi[0],hi[1])][:,2:]-R_latt),axis=1),np.linalg.norm((cv_dict['{:d}-{:d}'.format(hi[0],hi[1])][:,2:]-(1-R_latt)),axis=1))
        try: ##basis ordering won't necessarily be the same. 
            match = np.where(R_compare[0]<1e-4)[0] #only considering the first option--do I need the other?
            for mi in match:#find the match
                tmp_H = [*cv_dict['{:d}-{:d}'.format(hi[0],hi[1])][int(mi)][:2],*np.around(Rcv[ii],4),hi[-1]]

                H_new.append(tmp_H)

                if H_new[-1][0]>H_new[-1][1]:
                    H_new[-1] = H_conj(tmp_H)

        except IndexError:
           print('flag')
           continue

        
    print('Number of Bulk Hamiltonian Hopping Terms Found: {:d}, Number of Surface Basis Hopping Terms Filled: {:d}'.format(len(H_old),len(H_new)))
    
    Hobj = TB_lib.gen_H_obj(H_new)
    for h in Hobj:
        h.H = h.clean_H()
        
    return Hobj

def Hobj_to_dict(Hobj,basis):
    '''
    Associate a list of matrix elements with each orbital in the original basis. 
    Try something a little wien2k-style here. The hopping paths are given not as direct units, 
    but as number of unit-vectors for each hopping path. So the actual hopping path will be:
        np.dot(H[2:5],svec)+TB.basis[j].pos-TB.basis[i].pos
    This facilitates determining easily which basis element we are dealing with. For the slab,
    the new supercell will be extended along the 001 direction. So to redefine the orbital indices
    for a given element, we just take [i, len(basis)*(R_2)+j, (np.dot((R_0,R_1,R_2),svec)+pos[j]-pos[i]),H]
    If the path goes into the vacuum buffer don't add it to the new list!
    '''
    Hdict = {}
    for h in Hobj:
        try:
            Hdict[h.i] = Hdict[h.i] + [[h.i,h.j,*h.H[i]] for i in range(len(h.H))]
        except KeyError:
            Hdict[h.i] = [[h.i,h.j,*h.H[i]] for i in range(len(h.H))]
    return Hdict


def build_slab_H(Hsurf,slab_basis,surf_basis,svec):
    '''
    Build a slab Hamiltonian, having already defined the surface-unit cell Hamiltonian and basis.
    Begin by creating a dictionary corresponding to the Hamiltonian matrix elements associated with 
    the relevant surface unit cell orbital which pairs with our slab orbital, and all its possible hoppings
    in the original surface unit cell. This dictionary conveniently redefines the hopping paths in units
    of lattice vectors between the relevant orbitals. In this way, we can easily relabel a matrix element
    by the slab_basis elements, and then translate the connecting vector in terms of the pertinent orbitals.
    
    If the resulting element is from the lower diagonal, take its conjugate. Finally, only if the result is physical,
    i.e. corresponds to a hopping path contained in the slab, and not e.g. extending into the vacuum, should the matrix
    element be included in the new Hamiltonian. Finally, the new list Hnew is made into a Hamiltonian object, as always, and
    duplicates are removed.
    
    args:
        Hsurf -- Hamiltonian object from the surface unit cell
        slab_basis -- slab unit cell basis (list of orbital objects)
        surf_basis -- surface unit cell basis (list of orbital objects)
        svec -- surface unit cell lattice vectors (numpy array of 3x3 float)
    
    '''
    heights = np.array([o.pos[2] for o in slab_basis])
    limits = (heights.min(),heights.max())
    Hnew = []
    Hdict = Hobj_to_dict(Hsurf,surf_basis)
    si = np.linalg.inv(svec)
    for oi in slab_basis:
        Htmp = Hdict[oi.slab_index] ##### OK PROBLEM HERE!
        for hi in Htmp:
            ncells = np.floor(np.dot(hi[2:5],si))[2]
            Htmp_2 = [0]*6

            Htmp_2[0] = int(oi.index)
            Htmp_2[1] = int(oi.index/len(surf_basis))*len(surf_basis) + int(len(surf_basis)*ncells+hi[1])

            Htmp_2[5] = hi[5]
            
            try:
                Htmp_2[2:5] = hi[2:5]

                if 0<=Htmp_2[1]<len(slab_basis) and 0<=Htmp_2[0]<len(slab_basis):
                    if limits[0]<=(slab_basis[hi[0]].pos+Htmp_2[2:5])[2]<=limits[1]:
                        Hnew.append(Htmp_2)
            except IndexError:

                continue
    Hobj = TB_lib.gen_H_obj(Hnew)
    for h in Hobj:
        h.H = h.clean_H()
        
    return unpack(Hobj) #Modify to have function return list of H-elements
                


def bulk_to_slab(slab_dict):
    '''
    Wrapper function for generating a slab tight-binding model, having established a bulk model.
    args:
        slab_dict -- dictionary containing all essential information regarding the slab construction:
            'miller' -- miller indices as a numpy array len 3 of int
            'TB' -- Tight-binding model corresponding to the bulk model
            'fine' -- Fine adjustment of the slab limits, beyond the termination to precisely indicate the termination. float, units of Angstrom, relative to the bottom, and top surface generated
            'thick' -- minimum thickness of the slab structure (int)
            'vac' -- minimum thickness of the slab vacuum buffer to properly generate a surface with possible surface states (int)
            'termination' -- tuple of integers specifying the basis indices for the top and bottom of the slab structure
    return:
        tight-binding TB object containing the slab basis, slab Hamiltonian
    '''
    
    surf_basis,nvec,Rmat = gen_surface(slab_dict['TB'].avec,slab_dict['miller'],slab_dict['TB'].basis)
    surf_ham = H_surf(surf_basis,nvec,slab_dict['TB'].mat_els,Rmat)
    slab_vec,slab_basis = gen_slab(surf_basis,nvec,slab_dict['thick'],slab_dict['vac'],slab_dict['termination'],slab_dict['fine'])
    slab_ham = build_slab_H(surf_ham,slab_basis,surf_basis,nvec)
    
    slab_TB = slab_dict['TB'].copy()
    slab_TB.avec = slab_vec
    slab_TB.basis = slab_basis
 #   slab_TB.mat_els = slab_ham
    slab_TB.Kobj.kpts = np.dot(slab_dict['TB'].Kobj.kpts,Rmat)
    return slab_TB,slab_ham

           


def H_conj(h):
    return [h[1],h[0],-h[2],-h[3],-h[4],np.conj(h[5])]

def mod_dict(surf_basis,av_i):
    cv_dict = {}
    for bi in range(len(surf_basis)):
        for bj in range(len(surf_basis)):
            mod_vec = np.mod(np.around(np.dot((surf_basis[bj].pos-surf_basis[bi].pos),av_i),4),1)
#            mod_vec = np.around(np.dot(surf_basis[bj].pos-surf_basis[bi].pos,av_i),4)
            try:
                cv_dict['{:d}-{:d}'.format(surf_basis[bi].slab_index,surf_basis[bj].slab_index)].append([bi,bj,*mod_vec])
            except KeyError:
                cv_dict['{:d}-{:d}'.format(surf_basis[bi].slab_index,surf_basis[bj].slab_index)] =[[bi,bj,*mod_vec]]
    for cvi in cv_dict:

        cv_dict[cvi] = np.array(cv_dict[cvi])
    
    return cv_dict


if __name__ == "__main__":
    a = 3.606
    miller = np.array([1,1,1])
    avec = np.array([[a/2,a/2,0],[a/2,0,a/2],[0,a/2,a/2]]) 
    
    vn_b,R = basal_plane(v_vecs(miller,avec))
    

    