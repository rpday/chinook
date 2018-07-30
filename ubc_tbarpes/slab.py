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
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import ubc_tbarpes.v3_def as v3
import ubc_tbarpes.inside as inside
import ubc_tbarpes.orbital as olib
import ubc_tbarpes.TB_lib as TB_lib


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
    vp = np.around(np.dot(v,R),4)
    
    
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
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(vert[:,0],vert[:,1],vert[:,2])
    ax.scatter(box_pts[:,0],box_pts[:,1],box_pts[:,2],c='r',alpha=0.3)
    return vert,box_pts

def populate_box(box,basis,avec):
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
            tmp = b.pos + ri
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





#def Bi2Se3(a_vec):
#    nu = 0.792 #relative coordinates of the Bi-layers in the QL +/- (mu,mu,mu)
#    mu = 0.399 #same for Se1 -- +/-(nu,nu,nu)
#    atoms = [1,0,2,0,1] #list of atoms in unit cell--0 == Bismuth, 1 == Selenium(1), 2 == Selenium(2)
#    basis = []
#    pos = [-nu*(a_vec[0]+a_vec[1]+a_vec[2]),-mu*(a_vec[0]+a_vec[1]+a_vec[2]),np.array([0.0,0.0,0.0]),mu*(a_vec[0]+a_vec[1]+a_vec[2]),nu*(a_vec[0]+a_vec[1]+a_vec[2])] #orbital positions relative to origin
#    for p in list(enumerate(pos)):
#        basis.append(olib.orbital(p[1],atoms[p[0]],len(basis)))
#    return basis

def FeSe(avec):
    fpos = np.array([[0.25,0.75,0.0],[0.75,0.25,0.0],[0.25,0.25,0.2324],[0.75,0.75,0.7676]])
    pos = np.dot(fpos,avec)
    label = ['32xz','32xz','41z','41z']
    basis = []
    for p in list(enumerate(pos)):
        basis.append(olib.orbital(int(p[0]/2),len(basis),label[p[0]],p[1],26))
    return basis


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
    '''
    vn_b,R = basal_plane(v_vecs(miller,avec))
    pipe,box = par(vn_b)
    avec_R = np.dot(avec,R)
    b_points = populate_box(box,basis,avec_R)
    in_pped,inds = populate_par(b_points,vn_b)
    
    new_basis = np.empty(len(in_pped),dtype=olib.orbital)
    for ii in range(len(in_pped)):
        tmp = basis[int(inds[ii])].copy()
        tmp.slab_index = int(inds[ii])
        tmp.index = ii
        tmp.pos = in_pped[ii]
        new_basis[ii] = tmp
    
    
    return new_basis,vn_b
    


def gen_slab(basis,vn,mint,minb,term):
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
    return:
        avec - updated lattice vector for the SLAB unit cell, numpy array of float (3x3)
        cull - array of new orbital basis objects, with slab-index corresponding to the original basis indexing
    '''
    pts = []
    Nmin = int(np.ceil((mint+minb)/vn[2,2])+2)
    
    for i in range(Nmin):
        for bi in basis:
            pts.append([*(i*vn[-1]+bi.pos),bi.slab_index])
    pts = np.array(pts)
    pts = pts[pts[:,2].argsort()]
    term_1set = pts[pts[:,3]==term[1]]
    term_0set = pts[pts[:,3]==term[0]]
    surf = pts[:,2].max()-minb
    base = term_1set[:,2].min()
    top = term_0set[np.where(abs(term_0set[:,2]-surf)==abs(term_0set[:,2]-surf).min())[0][0],2]

    cull = np.array([pts[p] for p in range(len(pts)) if base<=pts[p,2]<=top])
    cull[:,2]-=top
    avec = np.copy(vn)
    avec[2] = Nmin*vn[2]
    
    new_basis = np.empty(len(cull),dtype=olib.orbital)
    for ii in range(len(cull)):
        tmp = basis[int(cull[ii,-1])].copy()
        tmp.slab_index = int(cull[ii,-1])
        tmp.index = ii
        tmp.pos = cull[ii,:3]
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
            if np.linalg.norm(el[:3])>0:
                Hlist.append([hij.j,hij.i,-el[0],-el[1],-el[2],np.conj(el[-1])])
    return Hlist
#
#
def H_surf(surf_basis,avec,H_bulk):
    '''
    Rewrite the bulk-Hamiltonian in terms of the surface unit cell, with its most likely expanded basis.
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
    H_new = np.empty((len(H_old),6),dtype=complex)
    for ii in range(len(H_old)):
        hi = H_old[ii]
        Rvec = hi[2:5]
        R_latt = np.mod(np.around(np.dot(Rvec,av_i),4),1)
        R_compare = (np.linalg.norm((cv_dict['{:d}-{:d}'.format(hi[0],hi[1])][:,2:]-R_latt),axis=1),np.linalg.norm((cv_dict['{:d}-{:d}'.format(hi[0],hi[1])][:,2:]-(1-R_latt)),axis=1))
#        print(R_compare)
        try: ##basis ordering won't necessarily be the same. 
            match = np.where(R_compare[0]<1e-4)[0][0]
            H_new[ii]=np.array([*cv_dict['{:d}-{:d}'.format(hi[0],hi[1])][int(match)][:2],*hi[2:]])

        except IndexError:
            match = np.where(R_compare[1]<1e-4)[0][0]
            H_new[ii] = np.array([*cv_dict['{:d}-{:d}'.format(hi[0],hi[1])][int(match)][:2],-hi[2],-hi[3],-hi[4],np.conj(hi[5])])
        if H_new[ii][0]>H_new[ii][1]:
            H_new[ii] = H_conj(H_new[ii])
    
    Hobj = TB_lib.gen_H_obj(H_new)
        
    return Hobj


def H_conj(h):
    return np.array([h[1],h[0],-h[2],-h[3],-h[4],np.conj(h[5])])

def mod_dict(surf_basis,av_i):
    cv_dict = {}
    for bi in range(len(surf_basis)):
        for bj in range(len(surf_basis)):
            mod_vec = np.mod(np.around(np.dot((surf_basis[bj].pos-surf_basis[bi].pos),av_i),4),1)
            try:
                cv_dict['{:d}-{:d}'.format(surf_basis[bi].slab_index,surf_basis[bj].slab_index)].append([bi,bj,*mod_vec])
            except KeyError:
                cv_dict['{:d}-{:d}'.format(surf_basis[bi].slab_index,surf_basis[bj].slab_index)] =[[bi,bj,*mod_vec]]
#            print(bi,bj,cv_dict['{:d}-{:d}'.format(surf_basis[bi].slab_index,surf_basis[bj].slab_index)])
    for cvi in cv_dict:

        cv_dict[cvi] = np.array(cv_dict[cvi])
    
    return cv_dict
    
    
    


if __name__=="__main__":
    #alpha-Fe2O3    
    avec = np.array([[0,4.736235,0],[4.101698,-2.368118,0],[0,0,13.492372]])
    miller = np.array([1,0,4])
    #Bi2Se3
#    alatt = 4.1141#4.1138
#    clatt = 28.64704#28.64 #using the QL model-
#    avec = np.array([[alatt/np.sqrt(3.0),0.0,clatt/3.0],[-alatt/(2.0*np.sqrt(3.0)),alatt/2.0,clatt/3.0],[-alatt/(2.0*np.sqrt(3.0)),-alatt/2.0,clatt/3.0]])
#    miller = np.array([1,1,1])
#    basis = Bi2Se3(avec)

    #FeSe
    avec = np.array([[3.25,3.25,0],[3.25,-3.25,0],[0,0,4]])
    miller = np.array([1,0,1])
#    
#    vn = v_vecs(miller,avec)
##    
#    vn_b,R = basal_plane(vn)
#
#    pipe,box = par(vn_b)
#    avec = np.dot(avec,R)
    basis = FeSe(avec)
    new_basis,vn_b = gen_surface(avec,miller,basis)
    H_surf = gen_H_surf(new_basis,H_bulk)
    nvec,cull=gen_slab(new_basis,vn_b,50,30,[2,3])

##    basis = FeO.basis(avec)
#    b_points = populate_box(box,basis,avec)
#    fig = plt.figure()
#    ax = fig.add_subplot(111,projection='3d')
#    ax.scatter(box[:,0],box[:,1],box[:,2])
#    ax.scatter(b_points[:,0],b_points[:,1],b_points[:,2])
#    
#    in_pped,inds = populate_par(b_points,vn_b)
#    fig = plt.figure()
#    ax = fig.add_subplot(111,projection='3d')
#    ax.scatter(pipe[:,0],pipe[:,1],pipe[:,2])
#    ax.scatter(in_pped[:,0],in_pped[:,1],in_pped[:,2],c=inds,cmap=cm.Spectral)
#    
#    build_slab(10,vn_b,in_pped)

#    near  = np.array([np.dot([int(np.mod(i,(3)))-1,int(np.mod(i/(3),(3)))-1,int(i/(3)**2)-1],vn_b) for i in range((3)**3)])
#    pts,crs = [],[]
#    for ni in near:
#        for b in list(enumerate(in_pped)):
#            pts.append(ni+b[1])
#            crs.append(cs[b[0]])
#    pts = np.array(pts)
#    crs = np.array(crs)
#    fig = plt.figure()
#    ax = fig.add_subplot(111,projection='3d')
#    ax.scatter(pts[:,0],pts[:,1],pts[:,2],c=inds,cmap=cm.RdBu)
            
#    cov_fpos,cov_labels = cov_transform(basis,inds,avec,vn_b)
#    
    



    