#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Created on Mon Nov 13 15:41:01 2017
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
import matplotlib.pyplot as plt
from operator import itemgetter
from itertools import compress
import sys

if sys.version_info<(3,0):
    print('Warning: This software requires Python 3.0 or higher. Please update your Python instance before proceeding')
else:
    import chinook.H_library as Hlib
    
try:
    import psutil
    ps_found = True
except ModuleNotFoundError:
    print('psutil not found, please load for better memory handling. See documentation for more detail')
    ps_found = False


'''
Tight-Binding Utility module

'''


class H_me:
    '''
    This class contains the relevant executables and data structure pertaining 
    to generation of the Hamiltonian matrix elements for a single set of 
    coupled basis orbitals. Its attributes include integer values 
    **i**, **j** indicating the basis indices, and a list of hopping
    vectors/matrix element values for the Hamiltonian. 
    
    The method **H2Hk** provides an executable function of momentum to allow
    broadcasting of the Hamiltonian over a large array of momenta.
    Python's flexible protocol for equivalency and passing variables by
    reference/value require definition of a copy operator which allows one to
    produce safely, a copy of the object rather than its coordinates 
    in memory alone.
    ***
    '''
    def __init__(self,i,j,executable=False):
        '''
        Initialize the H_me object, the related basis indices and an empty list
        of hopping elements.
        
        *args*:

            - **i**, **j**: int, basis indices

        *kwargs*:

            - **executable**: boolean, True if using unconventional, executable form of Hamiltonian rather
            than standard tij type tight-binding model
            
        ***
        '''
        self.executable = executable
        self.i = i
        self.j = j
        self.H = []
    
    def append_H(self,H,R0=0,R1=0,R2=0):
        '''
        Add a new hopping path to the coupling of the parent orbitals.
        
        *args*:

            - **H**: complex float, matrix element strength, or if self.exectype,
            should be an executable 
            
            - **R0**, **R1**, **R2**: float connecting vector in cartesian
            coordinate frame--this is the TOTAL vector, not the relevant 
            lattice vectors only
            
            
        *return*:

            - directly modifies the Hamiltonian list for these matrix
            coordinates

        ***
        '''
        if not self.executable:
            self.H.append([R0,R1,R2,H])
        else:
            self.H.append(H)
        
    def H2Hk(self):  
        '''
        Transform the list of hopping elements into a Fourier-series expansion 
        of the Hamiltonian. This is run during diagonalization for each
        matrix element index. If running a low-energy Hamiltonian, executable functions are
        simply summed for each basis index i,j, rather than computing a Fourier series. x is
        implicitly a numpy array of Nx3: it is essential that the executable conform to this input type.
        
        *return*:

            - lambda function of a numpy array of float of length 3
            
        ***
        '''
        if not self.executable:
            return lambda x: sum([complex(m[-1])*np.exp(1.0j*np.dot(x,np.array([m[0],m[1],m[2]]))) for m in self.H])
        return lambda x: sum([m(x) for m in self.H])
        
    def clean_H(self):
        
        '''
        
        Remove all duplicate instances of hopping elements in the matrix 
        element list. This function is run automatically during slab generation.
        
        The Hamiltonian list is not itself directly modified. 
        
        *return*:

            - list of hopping vectors and associated Hamiltonian matrix
            element strengths
            
        ***
        '''
        Hel_double = self.H
        bools = [True]*len(Hel_double)
        for hi in range(len(Hel_double)-1):
            for hj in range(hi+1,len(Hel_double)):
                
                try:
                    norm = np.linalg.norm(np.array(Hel_double[hi])-np.array(Hel_double[hj]))
                    if abs(norm)<1e-5:
                        bools[hj] = False
                except IndexError:
                    continue
        return list(compress(Hel_double,bools))
    
    def copy(self):
        
        '''
        Copy by value of the **H_me** object

        *return*:

            - **H_copy**: duplicate **H_me** object
       
        ***
        '''
        
        H_copy = H_me(self.i,self.j)
        H_copy.H = [h for h in self.H]
        return H_copy
    
    
        
class TB_model:
    '''
    The **TB_model** object carries the model basis as a list of **orbital**
    objects, as well as the model Hamiltonian, as a list of **H_me**. The orbital
    indices paired in each instance of **H_me** are stored in a dictionary under **ijpairs**
    '''
    
    def __init__(self,basis,H_args,Kobj = None):
        '''
        Initialize the tight-binding model object.
        
        *args*:  

            - **basis**: list of orbital objects
                    
            - **H_args**: dictionary for Hamiltonian build:
            
                - *'type'*: string,  "SK" ,"list", or "txt"
             
            - if *'type'* is *'SK'*:
                - *'V'*: dictionary of Slater-Koster hopping terms, 
                c.f. **chinook.H_library**
                
                - *'avec'*: numpy array of 3x3 float,
                lattice vectors for generating neighbours
                 
                
             - if *'type'* is *'list'*:
                 
                 - *'list'*: list of Hamiltonian elements: [i,j,R1,R2,R3,Hij]
                 
             - if *'type'* is 'txt':
                 
                 - *'filename'*:path to text file
                 
             - *'cutoff'*: float, cutoff distance for hopping
             
             - *'renorm'*: float, renormalization factor
                 
             - *'offset'*: float, offset energy, in eV
             
             - *'tol'*: float, minimum amplitude for matrix element term
            
             - *'spin'* : dictionary, with key values of 'bool':boolean,
             'soc':boolean, 'lam': dictionary of integer:float pairs
        
        *kwargs*:

            - **Kobj**: momentum object, as defined in *chinook.klib.py*
            
        ***
        '''
        self.basis = basis 
#        self.ijpairs = {}
        if H_args is not None:
            if 'avec' in H_args.keys():
                self.avec = H_args['avec']
        
            self.mat_els = self.build_ham(H_args)
            self.ijpairs = {(me[1].i,me[1].j):me[0] for me in list(enumerate(self.mat_els))}
            
        self.Kobj = Kobj

        
    def copy(self):
        '''
        Copy by value of the **TB_model** object
        
        *return*:

            - **TB_copy**: duplicate of the **TB_model** object.
        '''
        
        basis_copy = [o.copy() for o in self.basis]
        TB_copy = TB_model(basis_copy,None,self.Kobj)
        TB_copy.avec = self.avec.copy()
        TB_copy.ijpairs = self.ijpairs.copy()
        TB_copy.mat_els = [m.copy() for m in self.mat_els]
        
        return TB_copy
    
    def print_basis_summary(self):
        '''
        Very basic print function for printing a summary
        of the orbital basis, including their label, atomic species, position
        and spin character.
        '''
        print(' Index | Atom | Label | Spin |     Position     ')
        print('================================================')
        for o in self.basis:
            print('  {:3d}  |  {:2d}  |{:7}|{:6}| {:0.03f},{:0.03f},{:0.03f}'.format(o.index,o.atom,o.label,0.5*o.spin,*o.pos))
        
            
    
    def build_ham(self,H_args):
        
        '''
        Buld the Hamiltonian using functions from **chinook.H_library.py**

        *args*:

            - **H_args**: dictionary, containing all relevant information for
            defining the Hamiltonian list. For details, see **TB_model.__init__**.
            
        *return*:

            - sorted list of matrix element objects. These objects have 
            i,j attributes referencing the orbital basis indices, 
            and a list of form [R0,R1,R2,Re(H)+1.0jIm(H)]
        
        ***
        '''
        self.ijpairs = {}
        executable = False
        if type(H_args)==dict:
            try:
                ham_list = []
                if H_args['type'] == "SK":
                    ham_list = Hlib.sk_build(H_args['avec'],self.basis,H_args['V'],H_args['cutoff'],H_args['tol'],H_args['renorm'],H_args['offset'])
                elif H_args['type'] == "txt":
                    ham_list = Hlib.txt_build(H_args['filename'],H_args['cutoff'],H_args['renorm'],H_args['offset'],H_args['tol'])
                elif H_args['type'] == "list":
                    ham_list = H_args['list']
                elif H_args['type'] == 'exec':
                    ham_list = H_args['exec']
                    executable = True
                if 'spin' in H_args.keys():
                    if H_args['spin']['bool']:
                        ham_spin_double = Hlib.spin_double(ham_list,len(self.basis))
                        ham_list = ham_list + ham_spin_double   
                        if H_args['spin']['soc']:
                            ham_so = Hlib.SO(self.basis)
                            ham_list = ham_list + ham_so
                        if 'order' in H_args['spin']:
                            if H_args['spin']['order']=='F':
                                ham_FM = Hlib.FM_order(self.basis,H_args['spin']['dS'])
                                ham_list = ham_list + ham_FM
                            elif H_args['spin']['order']=='A':
                                ham_AF = Hlib.AFM_order(self.basis,H_args['spin']['dS'],H_args['spin']['p_up'],H_args['spin']['p_dn'])
                                ham_list = ham_list + ham_AF
                H_obj = gen_H_obj(ham_list,executable)
                return H_obj
            except KeyError:
                print('Invalid dictionary input for Hamiltonian generation.')
                return None
        else:
            return None
        
    def append_H(self,new_elements):
        '''
        Add new terms to the Hamiltonian by hand. This directly modifies
        the list of Hamiltonian matrix element, self.mat_els of the TB object.
        
        *args*:

            - **new_elements**: list of Hamiltonian matrix elements, either a single element [i,j,x_ij,y_ij,z_ij,H_ij(x,y,z)]
            or as a list of such lists. Here i, j are the related orbital-indices.
        
        ***
        '''
        if type(new_elements[0])!=list:
            new_elements = [new_elements]
        for elmts in new_elements:
            i,j = elmts[:2]
            if (i,j) in self.ijpairs.keys():
                self.mat_els[self.ijpairs[(i,j)]].H.append(elmts[2:])

            else:

                self.mat_els.append(H_me(i,j))
                self.mat_els[-1].append_H(R0 = elmts[2],R1 = elmts[3],R2 = elmts[4],H=elmts[5])
                self.ijpairs[(i,j)] = len(self.ijpairs.keys())
    def unpack(self):
        '''
        Reduce a Hamiltonian object down to a list of matrix elements. Include the Hermitian conjugate terms
    
    
        *return*:

            - **Hlist**: list of Hamiltonian matrix elements
            
        ***
        '''
        Hlist =[]
        for hij in self.mat_els:
            for el in hij.H:
                Hlist.append([hij.i,hij.j,*el])
        return Hlist
        
        
    def solve_H(self,Eonly = False):
        '''
        This function diagonalizes the Hamiltonian over an array of momentum vectors.
        It uses the **mat_el** objects to quickly define lambda functions of 
        momentum, which are then filled into the array and diagonalized.
        According to https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20050192421.pdf
        SVD algorithms require memory of 2*order*(4*order + 1) ~ 8*order^2. The matrices are
        complex float, so this should be 16 bytes per entry: so len(k)*(2048*order**2). If 
        the diagonalization is requesting more than 85% of the available memory, then split
        up the k-path into sequential diagonalizations.
        
        *return*:

            - **self.Eband**: numpy array of float, shape(len(self.Kobj.kpts),len(self.basis)),
            eigenvalues
            
            - **self.Evec**: numpy array of complex float, shape(len(self.Kobj.kpts),len(self.basis),len(self.basis))
            eigenvectors
                
        ***
        '''
        partition = False
        if ps_found:
            mem_summary = psutil.virtual_memory()
            avail = mem_summary.available
            size_lim = int(0.85*avail)
            mem_req = int(len(self.Kobj.kpts)*len(self.basis)**2*2048)
            if mem_req>size_lim:
                partition = True
                N_partitions = int(np.ceil(mem_req/size_lim))
                splits = [j*int(np.floor(len(self.Kobj.kpts)/N_partitions)) for j in range(N_partitions)]
                splits.append(len(self.Kobj.kpts))
                print('Large memory load: splitting diagonalization into {:d} segments'.format(N_partitions))
 
        if self.Kobj is not None:
            Hmat = np.zeros((len(self.Kobj.kpts),len(self.basis),len(self.basis)),dtype=complex) #initialize the Hamiltonian
            
            for me in self.mat_els:
                Hfunc = me.H2Hk() #transform the array above into a function of k
                Hmat[:,me.i,me.j] = Hfunc(self.Kobj.kpts) #populate the Hij for all k points defined
            if not partition:
                if not Eonly:
                    self.Eband,self.Evec = np.linalg.eigh(Hmat,UPLO='U') #diagonalize--my H_raw definition uses i<=j, so we want to use the upper triangle in diagonalizing
                else:
                    self.Eband = np.linalg.eigvalsh(Hmat,UPLO='U') #get eigenvalues only
                    self.Evec = np.array([0])
            else:
                self.Eband = np.zeros((len(self.Kobj.kpts),len(self.basis)))
                self.Evec =  np.zeros((len(self.Kobj.kpts),len(self.basis),len(self.basis)),dtype=complex)
                for ni in range(len(splits)-1):
                    self.Eband[splits[ni]:splits[ni+1]],self.Evec[splits[ni]:splits[ni+1]] = np.linalg.eigh(Hmat[splits[ni]:splits[ni+1]],UPLO ='U')
            return self.Eband,self.Evec
        else:
            print('You have not defined a set of kpoints over which to diagonalize.')
            return False
            
        
        
    def plotting(self,win_min=None,win_max=None,ax=None): #plots the band structure. Takes in Latex-format labels for the symmetry points indicated in the main code
        '''
        Plotting routine for a tight-binding model evaluated over some path in k.
        If the model has not yet been diagonalized, it is done automatically
        before proceeding.
        
        *kwargs*:

            - **win_min**, **win_max**: float, vertical axis limits for plotting
            in units of eV. If not passed, a reasonable choice is made which 
            covers the entire eigenspectrum.
        
            - **ax**: matplotlib Axes, for plotting on existing Axes
            
        *return*:

            - **ax**: matplotlib axes object
        
        ***
        '''
        try:
            Emin,Emax = np.amin(self.Eband),np.amax(self.Eband)
        except AttributeError:
            print('Bandstructure and energies have not yet been defined. Diagonalizing now.')
            self.solve_H()
            Emin,Emax = np.amin(self.Eband),np.amax(self.Eband)
        
        if ax is None:
            fig=plt.figure()
            fig.set_tight_layout(False)
            ax=fig.add_subplot(111)
            
        ax.axhline(y=0,color='k',lw=1.5,ls='--')
        for b in self.Kobj.kcut_brk:
            ax.axvline(x = b,color = 'k',ls='--',lw=1.5)
        for i in range(len(self.basis)):
            ax.plot(self.Kobj.kcut,np.transpose(self.Eband)[i,:],color='navy',lw=1.5)

        ax.set_xticks(self.Kobj.kcut_brk)
        ax.set_xticklabels(self.Kobj.labels)
        if win_max==None or win_min==None:
            ax.set_xlim(self.Kobj.kcut[0],self.Kobj.kcut[-1])
            ax.set_ylim(Emin-1.0,Emax+1.0)
        elif win_max !=None and win_min !=None:
            ax.set_xlim(self.Kobj.kcut[0],self.Kobj.kcut[-1])
            ax.set_ylim(win_min,win_max) 
        ax.set_ylabel("Energy (eV)")

        return ax  
    
    def plot_unitcell(self,ax=None):
        
        '''
        Utility script for visualizing the lattice and orbital basis.
        Distinct atoms are drawn in different colours
        
        *kwargs*:

            - **ax**: matplotlib Axes, for plotting on existing Axes
            
        *return*:

            - **ax**: matplotlib Axes, for further modifications to plot
        
        ***
        '''
        edges = cell_edges(self.avec)
        coord_dict = atom_coords(self.basis)

        
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        for ed in edges:
            ax.plot(ed[:,0],ed[:,1],ed[:,2],c='k')
        for atoms in coord_dict.keys():
            ax.scatter(coord_dict[atoms][:,0],coord_dict[atoms][:,1],coord_dict[atoms][:,2],s=20)
            
        return ax
          
            
def gen_H_obj(htmp,executable=False):
    
    '''
    Take a list of Hamiltonian matrix elements in list format:
    [i,j,Rij[0],Rij[1],Rij[2],Hij(R)] and generate a list of **H_me**
    objects instead. This collects all related matrix elements for a given
    orbital-pair for convenient generation of the matrix Hamiltonians over
    an input array of momentum
    
    *args*:

        - **htmp**: list of numeric-type values (mixed integer[:2], float[2:5], complex-float[-1])
    
    *kwargs*:

        - **executable**: boolean, if True, we don't have a standard Fourier-type Hamiltonian,
        but perhaps a low-energy expansion. In this case, the htmp elements are
        
    *return*:

        - **Hlist**: list of Hamiltonian matrix element, **H_me** objects
    
    ***
    '''
    if not executable:
        htmp = sorted(htmp,key=itemgetter(0,1,2,3,4))
    else:
        htmp = sorted(htmp,key=itemgetter(0,1))
    
    Hlist = []
    Hnow = H_me(int(htmp[0][0]),int(htmp[0][1]),executable=executable)
    Rij = np.zeros(3)
    
    for h in htmp:
        if h[0]!=Hnow.i or h[1]!=Hnow.j:
            Hlist.append(Hnow)
            Hnow = H_me(int(np.real(h[0])),int(np.real(h[1])),executable=executable)
        if not executable:
            Rij = np.real(h[2:5])
            Hnow.append_H(H=h[5],R0=Rij[0],R1=Rij[1],R2=Rij[2])
        else:
            Hnow.append_H(H=h[2])
    Hlist.append(Hnow)
    return Hlist 
            
        
def cell_edges(avec):

    '''
    Define set of line segments which enclose the unit cell. 
    
    *args*:

        - **avec**: numpy array of 3x3 float
        
    *return*:

        - **edges**: numpy array of 12 x 6, endpoints of the 12 edges of the unit cell parallelepiped
    
    ***
    '''
    
    modvec = np.array([[np.mod(int(j/4),2),np.mod(int(j/2),2),np.mod(j,2)] for j in range(8)])
    edges = []
    for p1 in range(len(modvec)):
        for p2 in range(p1,len(modvec)):
            if np.linalg.norm(modvec[p1]-modvec[p2])==1:
                edges.append([np.dot(avec.T,modvec[p1]),np.dot(avec.T,modvec[p2])])
    edges = np.array(edges)
    return edges


def atom_coords(basis):
    '''
    Define a dictionary organizing the distinct coordinates of instances of each
    atomic species in the basis
    
    *args*:

        - **basis**: list of orbital objects
        
    *return*:

        - **dictionary with integer keys, numpy array of float values. atom:locations are
        encoded in this way

    ***.
    '''
    
    coord_dict = {}
    all_pos = [[o.atom,*o.pos] for o in basis]
    for posns in all_pos:
        if posns[0] not in coord_dict.keys():
            coord_dict[posns[0]] = [np.array(posns[1:])]
        else:

            min_dist = np.array([np.sqrt((posns[1]-c[0])**2+(posns[2]-c[1])**2+(posns[3]-c[2])**2) for c in coord_dict[posns[0]]]).min()
            if min_dist>0:
                coord_dict[posns[0]].append(np.array(posns[1:]))
    for atoms in coord_dict:
        coord_dict[atoms] = np.array(coord_dict[atoms])
    return coord_dict
            
       
    
    
        
    
    
