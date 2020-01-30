#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Created on Thu Nov  9 21:38:24 2017

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
import numpy as np
import chinook.SlaterKoster as SK
import chinook.rotation_lib as rot_lib
import chinook.Ylm as Ylm


hb = 6.626*10**-34/(2*np.pi)
c  = 3.0*10**8
q = 1.602*10**-19
A = 10.0**-10
me = 9.11*10**-31
mn = 1.67*10**-27
kb = 1.38*10**-23



    

def txt_build(filename,cutoff,renorm,offset,tol):
    
    '''

    Build Hamiltonian from textfile, input is of form
    o1,o2,x12,y12,z12,t12, output in form [o1,o2,x12,y12,z12,t12]. 
    To be explicit, each row of the textfile is used to generate a
    k-space Hamiltonian matrix element of the form:

    .. math::
        H_{1,2}(k) = t_{1,2} e^{i (k_x x_{1,2} + k_y y_{1,2} + k_z z_{1,2})}

        
    *args*:

        - **filename**: string, name of file
        
        - **cutoff**: float, maximum distance of hopping allowed, Angstrom
        
        - **renorm**: float, renormalization of the bandstructure
        
        - **offset**: float, energy offset of chemical potential, electron volts
        
        - **tol**: float, minimum Hamiltonian matrix element amplitude
        
    *return*:

        - **Hlist**: the list of Hamiltonian matrix elements
    
    ***
    '''
    
    Hlist = []
    
    with open(filename,'r') as origin:
        
        for line in origin:
            
            spl = line.split(',')
            R = np.array([float(spl[2]),float(spl[3]),float(spl[4])])
            Hval = complex(spl[5])
            
            if len(spl)>6:
                Hval+=1.0j*float(spl[6])
            if abs(Hval)>tol and  np.linalg.norm(R)<cutoff:
                Hval*=renorm
                if np.linalg.norm(R)==0.0:
                    Hval-=offset
                    
                tmp = [int(spl[0]),int(spl[1]),R[0],R[1],R[2],Hval]

                Hlist.append(tmp)
            
    origin.close()
            
    return Hlist

def sk_build(avec,basis,Vdict,cutoff,tol,renorm,offset):
    
    '''

    Build SK model from using D-matrices, rather than a list of SK terms from table.
    This can handle orbitals of arbitrary orbital angular momentum in principal, 
    but right now implemented for up to and including f-electrons. 
    NOTE: f-hoppings require thorough testing
    
    *args*:

        - **avec**: numpy array 3x3 float, lattice vectors
        
        - **basis**: list of orbital objects
        
        - **Vdict**: dictionary, or list of dictionaries, of Slater-Koster integrals/ on-site energies
        
        - **cutoff**: float or list of float, indicating range where Vdict is applicable
        
        - **tol**: float, threshold value below which hoppings are neglected 
        
        - **offset**: float, offset for Fermi level 
        
    *return*:

        - **H_raw**: list of Hamiltonian matrix elements, in form [o1,o2,x12,y12,z12,t12]
    
    ***
    '''
        
    Vdict,cutoff,pts = cluster_init(Vdict,cutoff,avec) #build region of lattice points, containing at least the cutoff distance
    V = Vdict[0]
    if basis[0].spin!=basis[-1].spin: #only calculate explicitly for a single spin species
        brange = int(len(basis)/2)
    else:
        brange = len(basis)

    SK_matrices = SK.SK_full(basis[:brange]) #generate the generalized Slater-Koster matrices, as functions of R and potential V
    index_orbitals = index_ordering(basis[:brange]) #define the indices associated with the various orbital shells in the basis,
    H_raw = on_site(basis[:brange],V,offset) #fill in the on-site energies

    for i1 in index_orbitals:
        for i2 in index_orbitals:
            if index_orbitals[i1][index_orbitals[i1]>-1].min()<=index_orbitals[i2][index_orbitals[i2]>-1].min():
                o1o2 = (i1[0],i2[0],i1[1],i2[1],i1[2],i2[2])
                R12 = (np.array(i2[3:6])-np.array(i1[3:6])) 
                SKmat = SK_matrices[o1o2]
        
                for p in pts: #iterate over the points in the cluster
                    Rij = R12 + np.dot(p,avec)
                    Rijn = np.linalg.norm(Rij) #compute norm of the vector
#                    
                    if 0<Rijn<cutoff[-1]: #only proceed if within the cutoff distance
                        V = Vdict[np.where(Rijn>=cutoff)[0][-1]]
                    
                        Vlist = Vlist_gen(V,o1o2)
                        if Vlist is None:
                            continue
                        elif len(Vlist)==0:
                            continue
                        Euler_A,Euler_B,Euler_y = rot_lib.Euler(rot_lib.rotate_v1v2(Rij,np.array([0,0,1])))
                    
                        SKvals = mirror_SK([vi for vi in Vlist])
                        SKmat_num = SKmat(Euler_A,Euler_B,Euler_y,SKvals) #explicitly compute the relevant Hopping matrix for this vector and these shells
                        if abs(SKmat_num).max()>tol:

                            append = mat_els(Rij,SKmat_num,tol,index_orbitals[i1],index_orbitals[i2])
                            H_raw = H_raw + append
    return H_raw #finally return the list of Hamiltonian matrix elements




def on_site(basis,V,offset):
    
    '''

    On-site matrix element calculation. Try both anl and alabel formats,
    if neither is defined, default the onsite energy to 0.0 eV
    
    *args*:

        - **basis**: list of orbitals defining the tight-binding basis
        
        - **V**: dictionary, Slater Koster terms
        
        - **offset**: float, EF shift
        
    *return*:

        - **Ho**: list of Hamiltonian matrix elements
        
    ***
    '''
    Ho = []
    for oi in basis:
        try:
            H = V['{:d}{:d}{:d}'.format(oi.atom,oi.n,oi.l)]
        except KeyError:
            try:
                H = V['{:d}{:s}'.format(oi.atom,oi.label)]
            except KeyError:
                H = 0.0
        Ho.append([oi.index,oi.index,0.0,0.0,0.0,float(H-offset)])
    return Ho
    
                
def mat_els(Rij,SKmat,tol,i1,i2):
    '''

    Extract the pertinent, and non-zero elements of the Slater-Koster matrix
    and transform to the conventional form of Hamiltonian list entries
    (o1,o2,Rij0,Rij1,Rij2,H12(Rij))
    
    *args*:

        - **Rij**: numpy array of 3 float, relevant connecting vector 
        
        - **SKmat**: numpy array of float, matrix of hopping elements 
        for the coupling of two orbital shells 
        
        - **tol**: float, minimum hopping included in model
        
        - **i1**, **i2**: int,int, proper index ordering for the relevant
        instance of the orbital shells involved in hopping
        
    *return*:
        
        - **out**: list of Hamiltonian matrix elements, extracted from the
        ordered SKmat, in form [[o1,o2,x12,y12,z12,H12],...]
    
    ***    
    '''
    inds = np.where(abs(SKmat)>tol)
    out = []
    for ii in range(len(inds[0])):
        i_1 = i1[inds[0][ii]]
        i_2 = i2[inds[1][ii]]
        
        if -1<i_1<=i_2:
            out.append([i_1,i_2,*Rij,SKmat[inds[0][ii],inds[1][ii]]])

    return out
                    

def index_ordering(basis):
    
    '''

    We use an universal ordering convention for defining the Slater-Koster matrices
    which may (and most likely will) not match the ordering chosen by the user.
    To account for this, we define a dictionary which gives the ordering, relative 
    to the normal order convention defined here, associated with a given a-n-l shell
    at each site in the lattice basis.
    
    *args*:

        - **basis**: list of orbital objects
    
    *return*:

        - **indexing**: dictionary of key-value pairs (a,n,l,x,y,z):numpy.array([...])
    
    ***
    '''
    normal_order = {0:{'':0},1:{'x':0,'y':1,'z':2},2:{'xz':0,'yz':1,'xy':2,'ZR':3,'XY':4},3:{'z3':0,'xz2':1,'yz2':2,'xzy':3,'zXY':4,'xXY':5,'yXY':6}}
    indexing = {}
    for b in basis:
        anl = (b.atom,b.n,b.l,*np.around(b.pos,4))
        if anl not in indexing.keys():
            indexing[anl] = -1*np.ones(2*b.l+1)
        indexing[anl][normal_order[b.l][b.label[2:]]] = b.index
        
    return indexing

      

def Vlist_gen(V,pair):
   
    '''

    Select the relevant hopping matrix elements to be used in defining the value
    of the Slater-Koster matrix elements for a given pair of orbitals. Handles situation where
    insufficient parameters have been passed to system.
    
    *args*:

        - **V**: dictionary of Slater-Koster hopping terms
        
        - **pair**: tuple of int defining the orbitals to be paired, (a1,a2,n1,n2,l1,l2)
    
    *return*:
        
        - **Vvals**: numpy array of Vllx related to a given pairing, e.g. for s-p np.array([Vsps,Vspp])
    
    ***
    '''
    order = {'S':0,'P':1,'D':2,'F':3,0:'S',1:'P',2:'D',3:'F'}
    vstring = '{:d}{:d}{:d}{:d}{:d}{:d}'.format(*pair[:6])
    l = max(pair[4],pair[5])
    if len(V.keys())<(l+1):
        print('WARNING, insufficient number of Slater-Koster parameters passed: filling missing values with zeros.')
        for l_index in range(l+1):
            hopping_type = vstring+order[l_index]
            if hopping_type not in V.keys():
                V[hopping_type] = 0
    try:
        Vkeys = np.array(sorted([[l-order[vi[-1]],vi] for vi in V if vi[:-1]==vstring]))[:,1]
        Vvals = np.array([V[vk] for vk in Vkeys])
    except IndexError:
        vstring = '{:d}{:d}{:d}{:d}{:d}{:d}'.format(pair[1],pair[0],pair[3],pair[2],pair[5],pair[4])
        try: 
            Vkeys = np.array(sorted([[l-order[vi[-1]],vi] for vi in V if vi[:-1]==vstring]))[:,1]
            pre = (-1)**(pair[4]+pair[5]) #relative parity of the two coupled states
            Vvals = pre*np.array([V[vk] for vk in Vkeys])

        except IndexError:
            return None
    return Vvals
    
    
            
def mirror_SK(SK_in):
    
    '''
    
    Generate a list of values which is the input appended with its mirror 
    reflection. The mirror boundary condition suppresses the duplicate of the
    last value. e.g. [0,1,2,3,4] --> [0,1,2,3,4,3,2,1,0], 
    ['r','a','c','e','c','a','r'] --> ['r','a','c','e','c','a','r','a','c','e','c','a','r']
    Intended here to take an array of Slater-Koster hopping terms and reflect about 
    its last entry i.e. [Vsps,Vspp] -> [Vsps,Vspp,Vsps]
    
    *args*:

        - **SK_in**: iterable, of arbitrary length and data-type
        
    *return*:

        - list of values with same data-type as input
    
    ***
    '''
    return list(SK_in) + (SK_in[-2::-1])



def cluster_init(Vdict,cutoff,avec):
    
    '''
    Generate a cluster of neighbouring lattice points to use
    in defining the hopping paths--ensuring that it extends
    sufficiently far enough to capture even the largest hopping vectors.
    Also reforms the SK dictionary and cutoff lengths to be in list format.
    Returns an array of lattice points which go safely to the edge of the cutoff range.
    
    *args*:

        - **Vdict**: dictionary, or list of dictionaries of Slater Koster matrix elements
        
        - **cutoff**: float, or list of float
        
        - **avec**: numpy array of 3x3 float
        
    *return*:

        - **Vdict**: list of length 1 if a single dictionary passed, else unmodified
        
        - **cutoff**: numpy array, append 0 to the beginning of the cutoff list,
        else leave it alone.
        
        - **pts**: numpy array of lattice vector indices for a region of lattice points around
        the origin.
        
    ***
    '''
    if isinstance(cutoff,(int,float)) and not isinstance(cutoff,bool):
        cutoff = np.array([0.0,cutoff])
        Vdict = [Vdict]
    else:
        
    
        if cutoff[0]>0:
            cutoff.insert(0,0)
            cutoff = np.array(cutoff)
        else:
            cutoff = np.array(cutoff)
        

    pt_max = np.ceil(np.array([(cutoff).max()/np.linalg.norm(avec[i]) for i in range(len(avec))]).max())
    pts = region(int(pt_max)+1)
    return Vdict,cutoff,pts



###############################################################################
#########################Spin Orbit Coupling###################################
###############################################################################
    

def spin_double(H,lb):
    '''
    Duplicate the kinetic Hamiltonian terms to extend over the spin-duplicated 
    orbitals, which are by construction in same order and appended to end of the
    original basis.
    
    *args*:

        - **H**: list, Hamiltonian matrix elements [[o1,o2,x,y,z,H12],...]
        
        - **lb**: int, length of basis before spin duplication
        
    *return*:

        - **h2** modified copy of **H**, filled with kinetic terms for both 
        spin species
    
    ***
    '''
    lenb = int(lb/2)
    h2 = []
    for i in range(len(H)):
        h2.append([H[i][0]+lenb,H[i][1]+lenb,H[i][2],H[i][3],H[i][4],H[i][5]])
    return h2


def SO(basis):

    '''
    Generate L.S  matrix-elements for a given basis. 
    This is generic to all l, except the normal_order, which is defined here up to 
    and including the f electrons.
    Otherwise, this method is generic to any orbital angular momentum.
    
    In the factors dictionary defined here indicates the weight of the 
    different :math:`L_iS_i` terms. The keys are tuples of (L+/-/z,S+/-/z)
    in a bit of a cryptic way: for L, (0,1,2) ->(-1,0,1) and
    for S, (-1,0,1) = S1-S2 with S1,2 = +/- 1 here
    
    L+,L-,Lz matrices are defined for each l shell in the basis, 
    transformed into the basis of the tight-binding model.
    The nonzero terms will then just be used along with the spin and
    weighted by the factor value, and slotted into a len(**basis**)xlen(**basis**) matrix **HSO**
    
    *args*:

        - **basis**: list of orbital objects
    
    *return*:

        - **HSO**: list of matrix elements in standard format [o1,o2,0,0,0,H12]
        
    ***
    '''
    Md = Ylm.Yproj(basis)
    normal_order = {0:{'':0},1:{'x':0,'y':1,'z':2},2:{'xz':0,'yz':1,'xy':2,'ZR':3,'XY':4},3:{'z3':0,'xz2':1,'yz2':2,'xzy':3,'zXY':4,'xXY':5,'yXY':6}}
    factors = {(2,-1):0.5,(0,1):0.5,(1,0):1.0}
    L,al = {},[]
    HSO = []
    for o in basis[:int(len(basis)/2)]:
        if (o.atom,o.n,o.l) not in al:
            al.append((o.atom,o.n,o.l))
            Mdn = Md[(o.atom,o.n,o.l,-1)]
            Mup = Md[(o.atom,o.n,o.l,1)]
            Mdnp = np.linalg.inv(Mdn)
            Mupp = np.linalg.inv(Mup)
            L[(o.atom,o.n,o.l)] = [np.dot(Mupp,np.dot(Lm(o.l),Mdn)),np.dot(Mupp,np.dot(Lz(o.l),Mup)),np.dot(Mdnp,np.dot(Lp(o.l),Mup))]

    for o1 in basis:
        for o2 in basis:
            if o1.index<=o2.index:
                LS_val = 0.0
                if np.linalg.norm(o1.pos-o2.pos)<0.0001 and o1.l==o2.l and o1.n==o2.n:
                    inds = (normal_order[o1.l][o1.label[2:]],normal_order[o2.l][o2.label[2:]])
                    
                    ds = (o1.spin-o2.spin)/2.
                    if ds==0:
                        s=0.5*np.sign(o1.spin)
                    else:
                        s=1.0
                    for f in factors:
                        if f[1]==ds:
                            LS_val+=o1.lam*factors[f]*L[(o1.atom,o1.n,o1.l)][f[0]][inds]*s
                    HSO.append([o1.index,o2.index,0.,0.,0.,LS_val])

    return HSO


def Lp(l):

    '''
    L+ operator in the :math:`l`, :math:`m_l` basis, organized with 
    (0,0) = |l,l>... (2l,2l) = |l,-l>
    The nonzero elements are on the upper diagonal
    
    *arg*:

        - **l**: int orbital angular momentum
    
    *return*:

        - **M**: numpy array (2l+1,2l+1) of real float
        
    ***
    '''
    M = np.zeros((2*l+1,2*l+1))
    r = np.arange(0,2*l,1)
    M[r,r+1]=1.0
    vals = [0]+[np.sqrt(l*(l+1)-(l-m)*(l-m+1)) for m in range(1,2*l+1)]
    M = M*vals
    return M

def Lm(l):

    '''
    L- operator in the l,m_l basis, organized with 
    (0,0) = |l,l>... (2l,2l) = |l,-l>

    The nonzero elements are on the upper diagonal
    
    *arg*:

        - **l**: int orbital angular momentum
    
    *return*:

        - **M**: numpy array (2l+1,2l+1) of real float
        
    ***
    '''
    M = np.zeros((2*l+1,2*l+1))
    r = np.arange(1,2*l+1,1)
    M[r,r-1]=1.0
    vals = [np.sqrt(l*(l+1)-(l-m)*(l-m-1)) for m in range(0,2*l)]+[0]
    M = M*vals
    return M

def Lz(l):

    '''
    Lz operator in the l,:math:`m_l` basis
    
    *arg*:

        - **l**: int orbital angular momentum
    
    *return*:

        - numpy array (2*l+1,2*l+1)

    ***
    '''
    return np.identity(2*l+1)*np.array([l-m for m in range(2*l+1)])



def AFM_order(basis,dS,p_up,p_dn):
  
  '''
  Add antiferromagnetism to the tight-binding model, by adding a different on-site energy to 
  orbitals of different spin character, on the designated sites. 
  
  *args*:

      - **basis**: list, orbital objects
      
      - **dS**: float, size of spin-splitting (eV)
      
      - **p_up**, **p_dn**: numpy array of float indicating the orbital positions
      for the AFM order
      
  *return*: 

      - **h_AF**: list of matrix elements, as conventionally arranged [[o1,o2,0,0,0,H12],...]

  ***
  '''
  h_AF = []
  for bi in basis:
      if np.linalg.norm(bi.pos-p_up)==0:
          if bi.spin<0:
              h_AF.append([bi.index,bi.index,0,0,0,dS])
          else:
              h_AF.append([bi.index,bi.index,0,0,0,-dS])
      elif np.linalg.norm(bi.pos-p_dn)==0:
          if bi.spin<0:
              h_AF.append([bi.index,bi.index,0,0,0,-dS])
          else:
              h_AF.append([bi.index,bi.index,0,0,0,dS])
  return h_AF
    
    
def FM_order(basis,dS):
    
    '''

     Add ferromagnetism to the system. Take dS to assume that the splitting puts 
     spin-up lower in energy by dS,and viceversa for spin-down. This directly
     modifies the *TB_model*'s **mat_els** attribute
     
     *args*:

         - **basis**: list, of orbital objects in basis
         
         - **dS**: float, energy of the spin splitting (eV)
    
     *return*:

         - list of matrix elements [[o1,o2,0,0,0,H12],...]
    
     ***
     '''
    return [[bi.index,bi.index,0,0,0,-np.sign(bi.spin)*dS] for bi in basis]


#def Efield(basis,field,orbital_type='Slater'):
    
    '''
    Define a set of matrix elements which introduce an electric field, treated at the level of a dipole operator.
    
    TODO

    '''
 #   return None
    
    



def region(num):
    
    '''

    Generate a symmetric grid of points in number of lattice vectors. 
    
    *args*:

        - **num**: int, grid will have size 2*num+1 in each direction
    
    *return*:

        - numpy array of size ((2*num+1)**3,3) with centre value of first entry
        of (-num,-num,-num),...,(0,0,0),...,(num,num,num)

    ***
    '''
    num_symm = 2*num+1
    return np.array([[int(i/num_symm**2)-num,int(i/num_symm)%num_symm-num,i%num_symm-num] for i in range((num_symm)**3)])


