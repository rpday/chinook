# -*- coding: utf-8 -*-

#Created on Sat Jul 14 11:21:25 2018

#@author: rday

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




import numpy as np
from operator import itemgetter
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import chinook.rotation_lib as rotlib
import chinook.orbital as olib
import chinook.TB_lib as TB_lib
import chinook.surface_vector as surface_vector


def GCD(a,b):
    
    '''
    Basic greatest common denominator function. First find all divisors of each a and b.
    Then find the maximal common element of their divisors.
    
    *args*:

        - **a**, **b**: int
    
    *return*:

        - int, GCD of **a**, **b**
    
    ***
    '''
    dA,dB = divisors(a),divisors(b)
    return max(dA[i] for i in range(len(dA)) if dA[i] in dB)
    
def LCM(a,b):
    '''
    Basic lowest-common multiplier for two values a,b. Based on idea that LCM is just the
    product of the two input, divided by their greatest common denominator.
    
    *args*:

        - **a**, **b**: int
    
    *return*:

        - int, LCM of **a** and **b**
    
    ***
    '''
    return int(a*b/GCD(a,b))

def divisors(a):
    '''
    Iterate through all integer divisors of integer input
    
    *args*:

        - **a**: int
    
    *return*:

        list of int, divisors of **a**
        
    ***
    '''
    return [int(a/i) for i in range(1,a+1) if (a/i)%1==0]




def LCM_3(a,b,c):
    '''
    For generating spanning vectors, require lowest common multiple of 3
    integers, itself just the LCM of one of the numbers, and the LCM of the other two. 
    
    *args*:

        - **a**, **b**, **c**: int
        
    *return*:

        int, LCM of the three numbers
        
    ***
    '''
    
    return LCM(a,LCM(b,c))

def iszero(a):
    '''
    Find where an iterable of numeric is zero, returns empty list if none found
    
    *args*:

        - **a**: numpy array of numeric
        
    *return*:

        - list of int, indices of iterable where value is zero
        
    ***
    '''
    return np.where(a==0)[0]
    

def nonzero(a):
    '''
    Find where an iterable of numeric is non-zero, returns empty list if none found
    
    *args*:

        - **a**: numpy array of numeric
        
    *return*:

        - list of int, indices of iterable where value is non-zero
    
    ***
    '''
    return np.where(a!=0)[0]



def abs_to_frac(avec,vec):
    '''
    Quick function for taking a row-ordered matrix of lattice vectors: 
        | a_11  a_12  a_13  |
        | a_21  a_22  a_23  |
        | a_31  a_32  a_33  |
    and using it to transform a vector, written in absolute units, to fractional units.
    Note this function can be used to broadcast over N vectors you would like to transform
    
    *args*:

        - **avec**: numpy array of 3x3 float lattice vectors, ordered by rows
        
        - **vec**: numpy array of Nx3 float, vectors to be transformed to 
        fractional coordinates
        
     *return*:

         - Nx3 array of float, vectors translated into basis of lattice vectors    
    
    ***
    '''
    return np.dot(vec,np.linalg.inv(avec))

def frac_to_abs(avec,vec): 
    '''
    Same as abs_to_frac, but in opposite direction,from fractional to absolute coordinates
    
    *args*:

        - **avec**: numpy array of 3x3 float, lattice vectors, row-ordered 
        
        - **vec**: numpy array of Nx3 float, input vectors
        
    *return*:

        -  N x 3 array of float, vec in units of absolute coordinates (Angstrom)
    
    ***    
    '''
    return np.dot(vec,avec)


def p_vecs(miller,avec):
    '''
    Produce the vectors p, as defined by Ceder, to be used in defining spanning
    vectors for plane normal to the Miller axis
    
    *args*:

        - **miller**: numpy array of len 3 float
        
        - **avec**: numpy array of size 3x3 of float
    
    *return*:

        - **pvecs**: numpy array size 3x3 of float
    
    ***
    '''
    n_zero = nonzero(miller)
    zero = iszero(miller)
    pvecs = np.zeros((3,3))
    abs_miller = abs(miller)
    sgn_miller = np.sign(miller)
    if len(n_zero)==3:
        M = LCM_3(*abs_miller)
        M*=sgn_miller
        pvecs = np.array([avec[0]*M/miller[0],
                      avec[1]*M/miller[1],
                      avec[2]*M/miller[2]])
    elif len(n_zero)==2:
        M = LCM(*miller[n_zero])
        M = LCM_3(*abs_miller)
        M*=sgn_miller
        pvecs[n_zero[0]] = M/miller[n_zero[0]]*avec[n_zero[0]]
        pvecs[n_zero[1]] = M/miller[n_zero[1]]*avec[n_zero[1]]
        pvecs[zero[0]] = pvecs[n_zero[0]] + avec[zero[0]]
    elif len(n_zero)==1:
        pvecs[n_zero[0]] = np.zeros(3)
        pvecs[zero[0]] = avec[zero[0]]
        pvecs[zero[1]] = avec[zero[1]]
        
    return pvecs


def v_vecs(miller,avec):
    '''
    Wrapper for functions used to determine the vectors used to define the new,
    surface unit cell.

    *args*: 

        - **miller**: numpy array of 3 int, Miller indices for surface normal
        
        - **avec**: numpy array of 3x3 float, Lattice vectors
    
    *return*:

        - **vvecs**: new surface unit cell vectors numpy array of 3x3 float
    
    '''
    pvecs = p_vecs(miller,avec)
    vvecs = np.zeros((3,3))
    vvecs[0] = pvecs[1]-pvecs[0]
    vvecs[1] = pvecs[2]-pvecs[0]
    vvecs[2] = surface_vector.find_v3(vvecs[0],vvecs[1],avec,40)
    return vvecs

def basal_plane(vvecs):
    '''
    Everything is most convenient if we redefine the basal plane of the surface
    normal to be oriented within a Cartesian plane. To do so, we take the
    v-vectors. We get the norm of v1,v2 and then find the cross product with
    the z-axis, as well as the angle between these two vectors. We can then
    rotate the surface normal onto the z-axis.
    In this way we conveniently re-orient the v1,v2 axes into the Cartesian x,y plane.
    
    *args*:

        - **vvecs**: numpy array 3x3 float
    
    *return*:

        - **vvec_prime**: numpy array 3x3 of float, rotated v vectors
        
        - **Rmat**: numpy array of 3x3 float, rotation matrix to send original
        coordinate frame into the rotated coordinates.
        
    ***
    '''
    norm = np.cross(vvecs[0],vvecs[1])
    norm = norm/np.linalg.norm(norm)
    
    Rmat = rotlib.rotate_v1v2(norm,np.array([0,0,1])).T
    vvec_prime = np.dot(vvecs,Rmat)
    
    # Now perform one more rotation, taking vp[0] onto the [100] Cartesian axis
    
    phi = -np.arccos(vvec_prime[0,0]/np.linalg.norm(vvec_prime[0]))
    Rmat_prime = rotlib.Rodrigues_Rmat(np.array([0,0,1]),phi).T

    Rmat = Rmat@Rmat_prime #matrix multiplication of the two operators
    vvec_prime = np.dot(vvecs,Rmat)
    vvec_prime = np.around(vvec_prime,4)
    Rmat = np.around(Rmat,15)
    return vvec_prime,Rmat


def par(avec):
    '''
    Definition of the parallelepiped, as well as a containing region within the 
    Cartesian projection of this form which can then be used to guarantee correct
    definition of the new cell basis. The parallelipiped is generated, and then
    its extremal coordinates established, from which a containing parallelepiped is
    then defined.
    
    *args*:

        - **avec**: numpy array of 3x3 float
    
    *return*:

        - **vert**: numpy array  8x3 float vertices of parallelepiped
        
        - **box_pts**: numpy array 8 x 3 float vertices of containing box
    
    ***
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
    Populate the bounding box with points from the original lattice basis. These
    represent candidate orbitals to populate the surface-projected unit cell.
    
    *args*:

        - **box**: numpy array of 8x3 float, vertices of corner of a box
        
        - **basis**: list of orbital objects
        
        - **avec**: numpy array of 3x3 float, lattice vectors 
        
        - **R**: numpy array of 3x3 float, rotation matrix
        
    *return*:

        - **basis_full**: list of Nx4 float, representing instances of orbitals copies,
        retaining only their position and their orbital basis index. These orbitals fill
        a container box larger than the region of interest.
    
    ***
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
    Fill the box with basis points, keeping only those which reside in the new
    unit cell.
    
    *args*:

        - **points**: numpy array of Nx4 float ([:3] give position, [3] gives index)
        
        - **avec**: numpy array of  3x3 float
        
    *return*:

        - **new_points**: Nx3 numpy array of float, coordinates of new orbitals
        
        - **indices**: Nx1 numpy array of float, indices in original basis
    
    ***   
    '''

    in_points = frac_inside(points,avec)
    new_points = in_points[:,:3]
    indices = in_points[:,-1]

    return new_points,indices

def frac_inside(points,avec):
    '''
    Use fractional coordinates to determine whether a point is inside the new unit cell, or not.
    This is a very simple way of establishing this point, and circumvents many of the awkward 
    rounding issues of the parallelepiped method I have used previously. Ultimately however, 
    imprecision of the matrix multiplication and inversion result in some rounding error which
    must be corrected for. To do this, the fractional coordinates are rounded to the 4th digit.
    This leads to a smaller uncertainty by over an order to 10^3 than each rounding done on the 
    direct coordinates.
    
    *args*:

        - **points**: numpy array of float (Nx4) indicating positions and basis indices of the points to consider
        
        - **avec**: numpy array of 3x3 float, new lattice vectors
        
    *return*:

        - numpy array of Mx4 float, indicating positions and basis indices of the valid basis elements inside the new
        unit cell.
    
    ***
    '''
    fpoints = np.around(abs_to_frac(avec,points[:,:3]),4)
    bool_coords = np.array([True if (fp.min()>=0 and fp.max()<1) else False for fp in fpoints])
    return points[bool_coords]
    


def gen_surface(avec,miller,basis):
    '''
    Construct the surface unit cell, to then be propagated along the 001 direction to form a slab
    
    *args*:

        - **avec**: numpy array of 3x3 float, lattice vectors for original unit cell
        
        - **miller**: numpy array of 3 int, Miller indices indicating the surface orientation
        
        - **basis**: list of orbital objects, orbital basis for the original lattice
    
    *return*:

        - **new_basis**: list of orbitals, surface unit cell orbital basis
        
        - **vn_b**: numpy array of 3x3 float, the surface unit cell primitive lattice vectors
        
        - **Rmat**: numpy array of 3x3 float, rotation matrix, to be used in post-multiplication order
    
    ***
    '''
    vn_b,Rmat = basal_plane(v_vecs(miller,avec))
    pipe,box = par(vn_b)
    avec_R = np.dot(avec,Rmat)

    b_points = populate_box(box,basis,avec_R,Rmat)
    in_pped,inds = populate_par(b_points,vn_b)
    new_basis = np.empty(len(in_pped),dtype=olib.orbital)
    ordering = sorted_basis(in_pped,inds)

    for ii in range(len(in_pped)):
        tmp = basis[int(ordering[ii][3])].copy()
        tmp.slab_index = int(ordering[ii][3])
        tmp.index = ii
        tmp.pos = ordering[ii][:3]

        tmp.proj,tmp.Dmat = olib.rot_projection(tmp.l,tmp.proj,Rmat.T) #CHANGE TO TRANSPOSE

        new_basis[ii] = tmp
    
    
    return new_basis,vn_b,Rmat
    
def sorted_basis(pts,inds):
    '''
    Re-order the elements of the new basis, with preference to z-position followed
    by the original indexing
    
    *args*:

        - **pts**: numpy array of Nx3 float, orbital basis positions
        
        - **inds**: numpy array of N int, indices of orbitals, from original basis
        
    *return*:

        - **labels_sorted**: numpy array of Nx4 float, [x,y,z,index], in order of increasing z, and index

    ***    
    '''
    
    labels = np.array([[*pts[ii],inds[ii]] for ii in range(len(inds))])
    labels_sorted = np.array(sorted(labels,key=itemgetter(2,3)))
    return labels_sorted


def gen_slab(basis,vn,mint,minb,term,fine=(0,0)):
    '''
    Using the new basis defined for the surface unit cell, generate a slab
    of at least mint (minimum thickness), minb (minimum buffer) and terminated
    by orbital term. In principal the termination should be same on both top and
    bottom to avoid inversion symmetry breaking between the two lattice terminations.
    In certain cases, mint,minb may need to be tuned somewhat to get exactly the surface
    terminations you want.
    
    *args*:

        - **basis**: list of instances of orbital objects 
        
        - **vn**: numpy array of 3x3 float, surface unit cell lattice vectors 
        
        - **mint**: float, minimum thickness of the slab, in Angstrom
        
        - **minb**: float, minimum thickness of the vacuum buffer, in Angstrom
        
        - **term**: tuple of 2 int, termination of the slab tuple (term[0] = top termination, term[1] = bottom termination)
        
        - **fine**: tuple of 2 float, fine adjustment of the termination to precisely specify terminating atoms
        
    *return*:

        - **avec**: numpy array of float 3x3, updated lattice vector for the SLAB unit cell
        
        - **new_basis**: array of new orbital basis objects, with slab-index corresponding to the original basis indexing,
        and primary index corresponding to the order within the new slab basis
    
    ***
    '''
    pts = []
    Nmin = int(np.ceil((mint+minb)/vn[2,2])+2)
    
    for i in range(Nmin):
        for bi in basis:

            pts.append([*(i*vn[-1]+bi.pos),bi.atom,bi.slab_index,bi.index])
    pts = np.array(pts)
    z_order = pts[:,2].argsort()
    pts = pts[z_order]
    
    pts = np.array(sorted(pts,key=itemgetter(2,5)))
    #termination organized by ATOM
    term_1set = pts[pts[:,3]==term[1]]
    term_0set = pts[pts[:,3]==term[0]]
    surf = pts[:,2].max()-minb

    base = term_1set[term_1set[:,2]>=(term_1set[:,2].min()+fine[1]),2].min()

    top = term_0set[np.where(abs(term_0set[(term_0set[:,2]-surf)<fine[0],2]-surf)==abs(term_0set[(term_0set[:,2]-surf)<fine[0],2]-surf).min())[0][0],2]

    cull = np.array([pts[p] for p in range(len(pts)) if base<=pts[p,2]<=top])
    cull[:,2]-=top
    avec = np.copy(vn)
    avec[2] = Nmin*vn[2]
    
    new_basis = np.empty(len(cull),dtype=olib.orbital)
    for ii in range(len(cull)):
        iter_orbital = basis[int(cull[ii,-1])].copy()
        iter_orbital.slab_index = int(cull[ii,-1])
        iter_orbital.index = ii
        iter_orbital.pos = cull[ii,:3]-np.array([cull[-1,0],cull[-1,1],0])
        iter_orbital.depth = iter_orbital.pos[2]
        new_basis[ii] = iter_orbital
        
    return avec,new_basis


def region(num):
    '''
    Generate a symmetric grid of points in number of lattice vectors. 
    
    *args*:

        - **num**: int, grid will have size 2 num+1 in each direction
    
    *return*:

        - numpy array of size ((2 num+1)^3,3) with centre value of first entry of
        (-num,-num,-num),...,(0,0,0),...,(num,num,num)
   
    ***
    '''
    num_symm = 2*num+1
    return np.array([[int(i/num_symm**2)-num,int(i/num_symm)%num_symm-num,i%num_symm-num] for i in range((num_symm)**3)])


def unpack(Ham_obj):
    '''
    Reduce a Hamiltonian object down to a list of matrix elements. Include the Hermitian conjugate terms

    *args*:

        - **Ham_obj**: Hamiltonian object, c.f. *chinook.TB_lib.H_me*

    *return*:

        - **Hlist**: list of Hamiltonian matrix elements
        
    ***
    '''
    Hlist =[]
    for hij in Ham_obj:
        for el in hij.H:
            Hlist.append([hij.i,hij.j,*el])
    return Hlist
#
#
def H_surf(surf_basis,avec,H_bulk,Rmat,lenbasis):
    '''
    Rewrite the bulk-Hamiltonian in terms of the surface unit cell, with its
    (most likely expanded) basis. The idea here is to organize all 'duplicate' 
    orbitals, in terms of their various connecting vectors. Using modular
    arithmetic, we then create an organized dictionary which categorizes the 
    hopping paths within the new unit cell according to the new basis index
    designation. For each element in the Hamiltonian then, we can do the same
    modular definition of the hopping vector, easily determining which orbital
    in our new basis this hopping path indeed corresponds to. We then make a
    new list, organizing corresponding to the new basis listing.
    
    *args*:

        - **surf_basis**: list of orbitals in the surface unit cell
        
        - **avec**: numpy array 3x3 of float, surface unit cell vectors
        
        - **H_bulk**: *H_me* object(defined in *chinook.TB_lib.py*), as 
        the bulk-Hamiltonian
        
        - **Rmat**: 3x3 numpy array of float, rotation matrix 
        (pre-multiply vectors) for rotating the coordinate system from bulk 
        to surface unit cell axes
        
        - **lenbasis**: int, length of bulk basis
        
    *return*:

        - Hamiltonian object, written in the basis of the surface unit cell,
        and its coordinate frame, rather than those of the bulk system
        
    ***
    '''
    av_i = np.linalg.inv(avec) #inverse lattice vectors
    cv_dict = mod_dict(surf_basis,av_i) #generate dictionary of connecting vectors between each of the relevant orbitals, according to their definition in the bulk lattice.

    H_old = unpack(H_bulk) #transform Hamiltonian object to list of hopping terms

    Rcv = np.dot(np.array([h[2:5] for h in H_old]),Rmat) #rotate all hopping paths into the coordinate frame of the surface unit cell
    H_new = [] #initialize the new Hamiltonian list
    for ii in range(len(H_old)): #iterate over all original Hamiltonian hopping paths
        hi = H_old[ii] #temporary Hamiltonian matrix element to consider
        R_latt = np.mod(np.around(np.dot(Rcv[ii],av_i),3),1) #what is the hopping path, in new coordinate frame, in terms of modular vector (mod lattice vector)
        R_compare = np.linalg.norm(R_latt-(cv_dict['{:d}-{:d}'.format(hi[0],hi[1])][:,2:5]),axis=1)#,np.linalg.norm((cv_dict['{:d}-{:d}'.format(hi[0],hi[1])][:,2:]-(1-R_latt)),axis=1)) #two possible choices: 

        try:
            
            match = np.where(R_compare<5e-4)[0]


            for mi in match:#find the match
                tmp_H = [*cv_dict['{:d}-{:d}'.format(hi[0],hi[1])][int(mi)][:2],*np.around(Rcv[ii],4),hi[-1]]

                H_new.append(tmp_H)

                if H_new[-1][0]>H_new[-1][1]:
                    H_new[-1] = H_conj(tmp_H)


        except IndexError:
           print('ERROR: no valid hopping path found relating original Hamiltonian to surface unit cell.')
           continue

        
    print('Number of Bulk Hamiltonian Hopping Terms Found: {:d}, Number of Surface Basis Hopping Terms Filled: {:d}'.format(len(H_old),len(H_new)))
    if (len(H_new)/len(H_old))!=(len(surf_basis)/(lenbasis)):
        print('Going from {:d} to {:d} basis states'.format(lenbasis,len(surf_basis)))
        print('Invalid HAMILTONIAN! Missing hopping paths.')
        return []
    
    Hobj = TB_lib.gen_H_obj(H_new)

        
    return Hobj

def Hobj_to_dict(Hobj,basis):
   
    '''
    Associate a list of matrix elements with each orbital in the original basis. 
    The hopping paths are given not as direct units,but as number of unit-vectors
    for each hopping path. So the actual hopping path will be:
        np.dot(H[2:5],svec)+TB.basis[j].pos-TB.basis[i].pos
    
    This facilitates determining easily which basis element we are dealing with.
    For the slab, the new supercell will be extended along the 001 direction. 
    So to redefine the orbital indices for a given element, we just take 
    [i, len(basis)*(R_2)+j, (np.dot((R_0,R_1,R_2),svec)+pos[j]-pos[i]),H]
    If the path goes into the vacuum buffer don't add it to the new list!
    
    *args*:

        - **Hobj**: *H_me* object(defined in *chinook.TB_lib.py*), as 
        the bulk-Hamiltonian
        
        - **basis**: list of *orbital* objects
    
    *return*:

        - **Hdict**: dictionary of hopping paths associated with a given orbital
        index
    
    ***
    '''
    Hdict = {ii:[] for ii in range(len(basis))}
    Hlist = unpack(Hobj)
    
    for hi in Hlist:
        Hdict[hi[0]].append([*hi])
        Hdict[hi[1]].append([*H_conj(hi)])
    
    return Hdict
    

def build_slab_H(Hsurf,slab_basis,surf_basis,svec):
    '''
    Build a slab Hamiltonian, having already defined the surface-unit cell
    Hamiltonian and basis. Begin by creating a dictionary corresponding to the
    Hamiltonian matrix elements associated with the relevant surface unit cell 
    orbital which pairs with our slab orbital, and all its possible hoppings
    in the original surface unit cell. This dictionary conveniently redefines
    the hopping paths in units of lattice vectors between the relevant orbitals.
    In this way, we can easily relabel a matrix element by the slab_basis 
    elements, and then translate the connecting vector in terms of the 
    pertinent orbitals.
    
    If the resulting element is from the lower diagonal, take its conjugate.
    Finally, only if the result is physical, i.e. corresponds to a hopping path
    contained in the slab, and not e.g. extending into the vacuum, 
    should the matrix element be included in the new Hamiltonian. Finally,
    the new list Hnew is made into a Hamiltonian object, as always, and
    duplicates are removed.
    
    *args*:

        - **Hsurf**: *H_me* object(defined in *chinook.TB_lib.py*), as 
        the bulk-Hamiltonian from the surface unit cell
        
        - **slab_basis**: list of orbital objects, slab unit cell basis 
        
        - **surf_basis**: list of orbital objects, surface unit cell basis 
        
        - **svec**: numpy array of 3x3 float, surface unit cell lattice vectors 
        
    *return*:

        - list of Hamiltonian matrix elements in [i,j,x,y,z,Hij] format
        
    ***
    '''
    Hnew = []
    Hdict = Hobj_to_dict(Hsurf,surf_basis) #dictionary of hoppings. keys correspond to the slab_index, values the relative hoppings elements
    si = np.linalg.inv(svec)
    D = slab_basis[0].slab_index
    for oi in slab_basis:
        Htmp = Hdict[oi.slab_index] #access relevant hopping paths for the orbital in question
        for hi in Htmp: #iterate over all relevant hoppings
            
            ncells = int(np.round(np.dot(hi[2:5]-surf_basis[hi[1]].pos+surf_basis[hi[0]].pos,si)[2])) #how many unit cells -- in the surface unit cell basis are jumped during this hopping--specifically, cells along the normal direction

            Htmp_2 = [0]*6 #create empty hopping element, to be filled

            Htmp_2[0] = int(oi.index) #matrix row should be the SLAB BASIS INDEX
            Htmp_2[1] = int((D+oi.index)/len(surf_basis))*len(surf_basis) + int(len(surf_basis)*ncells+hi[1]-D)
            #matrix column is calculated as follows:
            #the first orbital's slab index is often not zero, D is the place-holder for the actual start. Following this
            # index increments monotonically, while slab_index is defined mod-len(surf_basis). To get the new 'j' index,
            #we find first the 'surface-cell' number of 'i', defined as int((D+i)/len(surf))*len(surf). Then we increment
            #by the integer number of surface-unit cells covered by the hopping vector, and further by the difference between
            #the original o2 index j, and the starting slab_index D.

            Htmp_2[5] = hi[5]       
            try:
                Htmp_2[2:5] = hi[2:5]

                if 0<=Htmp_2[1]<len(slab_basis) and 0<=Htmp_2[0]<len(slab_basis):
                    if Htmp_2[1]>=Htmp_2[0]:

                        Hnew.append(Htmp_2)  

            except IndexError:

                continue
    Hobj = TB_lib.gen_H_obj(Hnew)
    print('clean Hamiltonian')
    for h in Hobj:
        tmp_H = h.clean_H()

        h.H = tmp_H.copy()

        
    return unpack(Hobj) 
                


def bulk_to_slab(slab_dict):
    '''
    Wrapper function for generating a slab tight-binding model, having 
    established a bulk model.
    
    *args*:

        - **slab_dict**: dictionary containing all essential information
        regarding the slab construction:
        
            - *'miller'*: numpy array len 3 of int, miller indices 
            
            - *'TB'*: Tight-binding model corresponding to the bulk model
            
            - *'fine'*:  tuple of 2 float. Fine adjustment of the slab limits, 
            beyond the termination to precisely indicate the termination. 
            units of Angstrom, relative to the bottom, and top surface generated
            
            - *'thick'*: float, minimum thickness of the slab structure
            
            - *'vac'*: float, minimum thickness of the slab vacuum buffer
            to properly generate a surface with possible surface states
            
            - *'termination'*: tuple of 2 int, specifying the basis indices
            for the top and bottom of the slab structure
            
    *return*:

        - **slab_TB**: tight-binding TB object containing the slab basis
        
        - **slab_ham**: Hamiltonian object, slab Hamiltonian
        
        - **Rmat**: numpy array of 3x3 float, rotation matrix
        
    ***
    '''
    
    surf_basis,nvec,Rmat = gen_surface(slab_dict['TB'].avec,slab_dict['miller'],slab_dict['TB'].basis)
    surf_ham = H_surf(surf_basis,nvec,slab_dict['TB'].mat_els,Rmat,len(slab_dict['TB'].basis))
    slab_vec,slab_basis = gen_slab(surf_basis,nvec,slab_dict['thick'],slab_dict['vac'],slab_dict['termination'],slab_dict['fine'])
    slab_ham = build_slab_H(surf_ham,slab_basis,surf_basis,nvec)
    
    slab_TB = slab_dict['TB'].copy()
    slab_TB.avec = slab_vec
    slab_TB.basis = slab_basis
    slab_TB.Kobj.kpts = np.dot(slab_dict['TB'].Kobj.kpts,Rmat)
    return slab_TB,slab_ham,Rmat

          

def H_conj(h):
    '''
    Conjugate hopping path
    
    *args*:

        - **h**: list, input hopping path in format [i,j,x,y,z,Hij]
    
    *return*:

        - list, reversed hopping path, swapped indices, complex conjugate of the
        hopping strength
    
    ***
    '''
    return [h[1],h[0],-h[2],-h[3],-h[4],np.conj(h[5])]

def mod_dict(surf_basis,av_i):
    '''
    Define dictionary establishing connection between slab basis elements and the 
    bulk Hamiltonian. The slab_indices relate to the bulk model, we can then compile
    a list of *slab* orbital pairs (along with their connecting vectors) which should
    be related to a given bulk model hopping. The hopping is expressed in terms of the
    number of surface lattice vectors, rather than direct units of Angstrom. 
    
    *args*:

        - **surf_basis**: list of orbital objects, covering the slab model
        
        - **av_i**: numpy array of 3x3 float, inverse of the lattice vector matrix
        
    *return*:
    
        - **cv_dict**: dictionary with key-value pairs of 
        slab_index[i]-slab_index[j]:numpy.array([[i,j,mod_vec]...])
            
    ***
    '''
    cv_dict = {}
    for bi in range(len(surf_basis)):
        for bj in range(len(surf_basis)):
            mod_vec = np.mod(np.around(np.dot((surf_basis[bj].pos-surf_basis[bi].pos),av_i),3),1)


            try:
                cv_dict['{:d}-{:d}'.format(surf_basis[bi].slab_index,surf_basis[bj].slab_index)].append([bi,bj,*mod_vec])
            except KeyError:
                cv_dict['{:d}-{:d}'.format(surf_basis[bi].slab_index,surf_basis[bj].slab_index)] =[[bi,bj,*mod_vec]]
    for cvi in cv_dict:

        cv_dict[cvi] = np.array(cv_dict[cvi])
    
    return cv_dict


    