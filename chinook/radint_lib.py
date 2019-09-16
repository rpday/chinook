# -*- coding: utf-8 -*-

#Created on Tue Jan 29 16:02:40 2019

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
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import chinook.adaptive_int as adint
import chinook.electron_configs as econ

####PHYSICAL CONSTANTS RELEVANT TO CALCULATION#######
hb = 6.626*10**-34/(2*np.pi)
c  = 3.0*10**8
q = 1.602*10**-19
A = 10.0**-10
me = 9.11*10**-31
mN = 1.67*10**-27
kb = 1.38*10**-23


def make_radint_pointer(rad_dict,basis,Eb):
    
    '''
    Define executable radial integral functions, and store in a 
    pointer-integer referenced array. This allows for fewer executions
    of the interpolation function in the event where several orbitals
    in the basis share the same a,n,l. Each of these gets 2 functions
    for l +/-1, which are stored in the rows of the array **B_array**.
    The orbitals in the basis then are matched to these executables,
    with the corresponding executable row index saved in **B_pointers**.
    
    Begin by defining the executable radial wavefunctions, then perform
    integration at several binding energies, finally returning an
    interpolation of these integrations.
    
    *args*: 

        - **rad_dict**: dictionary of ARPES parameters: relevant keys are 
        'hv' (photon energy), 'W' (work function), and the rad_type
        (radial wavefunction type, as well as any relevant additional
        pars, c.f. *radint_lib.define_radial_wavefunctions*).
        Note: *'rad_type'* is optional, (as is *rad_args*, depending on choice
        of radial wavefunction.)
        
        - **basis**: list of orbitals in the basis
        
        - **Eb**: tuple of 2 floats indicating the range of energy of
        interest (increasing order)
        
    *return*: 

        - **B_array**: numpy array of Nx2 executable functions of float
        
        - **B_pointers**: numpy array of integer indices matching orbital
        basis ordering to the functions in **B_array**
        
    ***   
    '''
    radial_funcs = define_radial_wavefunctions(rad_dict,basis)
    fixed = True if ('rad_type' in rad_dict.keys() and rad_dict['rad_type']=='fixed') else False
    B_dictionary = fill_radint_dic(Eb,radial_funcs,rad_dict['hv'],rad_dict['W'],rad_dict['phase_shifts'],fixed)
    B_array,B_pointers = radint_dict_to_arr(B_dictionary,basis)
    
    return B_array,B_pointers


def radint_dict_to_arr(Bdict,basis):

    '''
    Take a dictionary of executables defined for different combinations
    of a,n,l and send them to an array, with a corresponding pointer
    array which can be used to dereference the relevant executable.
    
    *args*:     

        - **Bdict**: dictionary of executables with 'a-n-l-l'' keys
        
        - **basis**: list of orbital objects
    
    *return*:  

        - **Blist**: numpy array of the executables, organized by a-n-l,
        and l' (size Nx2, where N is the length of the set of 
        distinct a-n-l triplets)
        
        - **pointers**: numpy array of length (basis), integer datatype
        indicating the related positions in the **Blist** array
    
    ***
    '''
    Blist = []
    lchan = []
    pointers = []
    for b in basis:
        orbstr = '{:d}-{:d}-{:d}'.format(b.atom,b.n,b.l)
        if orbstr not in lchan:
            lchan.append(orbstr)
            Blist.append([Bdict[orbstr+'-{:d}'.format(b.l-1)],Bdict[orbstr+'-{:d}'.format(b.l+1)]])
        pointers.append(lchan.index(orbstr))
    return np.array(Blist),np.array(pointers)
        


def define_radial_wavefunctions(rad_dict,basis):
    
    '''
    Define the executable radial wavefunctions for computation of
    the radial integrals
    
    *args*:   

        - **rad_dict**: essential key is *'rad_type'*, if not passed,
        assume Slater orbitals. 
            
        - **rad_dict['rad_type']**:

                - *'slater'*: default value, if *'rad_type'* is not passed,
                Slater type orbitals assumed and evaluated for the integral
                    - *'rad_args'*: dictionary of float, supplying optional final-state 
                    phase shifts, accounting for scattering-type final states. keys of form
                    'a-n-l-lp'. Radial integrals will be accordingly multiplied
                
                - *'hydrogenic'*: similar in execution to *'slater'*, 
                but uses Hydrogenic orbitals--more realistic for light-atoms
                    - *'rad_args'*: dictionary of float, supplying optional final-state 
                    phase shifts, accounting for scattering-type final states. keys of form
                    'a-n-l-lp'. Radial integrals will be accordingly multiplied
                
                - *'grid'*: radial wavefunctions evaluated on a grid of
                radii. Requires also another key_value pair:
                    
                   - *'rad_args'*: dictionary of numpy arrays evaluating
                   the radial wavefunctions. Requires an *'r'* array,
                   as well as 'a-n-l' indicating 'atom-principal quantum number-orbital angular momentum'.
                   Must pass such a grid for each orbital in the basis!
                   
                - *'exec'*: executable functions for each 'a-n-l' i.e.
                'atom-principal quantum number-orbital angular momentum'. 
                If 	executable is chosen, require also:
                    
                        - *'rad_args'*, which will be a dictionary of
                        executables, labelled by the keys 'a-n-l'. 
                        These will be passed to the integral routine.
                        Note that it is required that the executables
                        are localized, i.e. vanishing for large radii.
                        
                - *'fixed*: radial integrals taken to be constant float,
                require dictionary:
                    
                    - *'rad_args'* with keys 'a-n-l-lp', i.e. 
                    'atom-principal quantum number-orbital angular momentum-final state angular momentum'
                    and complex float values for the radial integrals.
                    
        - **basis**: list of orbital objects
        
    *return*: 

        **orbital_funcs**: dictionary of executables
        
    ***    
    '''
    
    orbital_funcs = {}
    orbitals = gen_orb_labels(basis)

    if 'rad_type' not in rad_dict.keys():
        rad_dict['rad_type'] = 'slater'
    
    if rad_dict['rad_type'].lower()=='slater':
        for o in orbitals:
            orbital_funcs[o] = econ.Slater_exec(*orbitals[o])
    
    elif rad_dict['rad_type'].lower() == 'hydrogenic':
        for o in orbitals:
            orbital_funcs[o] = econ.hydrogenic_exec(*orbitals[o])
        
    elif rad_dict['rad_type'].lower() == 'grid':
        if 'rad_args' not in rad_dict.keys():
            print('ERROR: No "rad_args" key passing the grid values passed to ARPES calculation.\n Exiting. \n See Radial Integrals in the Manual for further details.\n')
            return None
        elif np.sum([o in rad_dict['rad_args'].keys() for o in orbitals])<len(orbitals):
            print('ERROR: Missing radial wavefunction grids--confirm all atoms and orbital shells have a grid.\n Exiting.\n')
            return None
        elif 'r' not in rad_dict['rad_args'].keys():
            print('ERROR: no radial grid passed for proper scaling of the radial wavefunctions.\n Exiting.\n')
            return None
        if rad_dict['rad_args']['r'].min()>0:
            print('ERROR: radial grid must go to the origin.\n Exiting.\n')
            return None
        else:
            for o in orbitals:
                orbital_funcs[o] = interp1d(rad_dict['rad_args']['r'],rad_dict['rad_args'][o])
    
    elif rad_dict['rad_type'].lower() == 'exec':
        if 'rad_args' not in rad_dict.keys():
            print('ERROR: No "rad_args" key passing an executable to ARPES calculation.\n. Exiting.\n See Radial Integrals in the documentation for further details.\n')
            return None
        elif np.sum([o in rad_dict['rad_args'].keys() for o in orbitals])<len(orbitals):
            print('ERROR: Missing radial wavefunction functionss--confirm all atoms and orbital shells have a function.\n Exiting.\n')
            return None
        else:
            for o in orbitals:
                orbital_funcs[o] = rad_dict['rad_args'][o]
    
    
    elif rad_dict['rad_type'].lower() == 'fixed':    
        if 'rad_args' not in rad_dict.keys():
            print('ERROR: Missing radial integral values.\n Exiting.\n See Radial Integrals in the documentation for further details.\n')
            return None
        for o in basis:
            lp = np.array([o.l-1,o.l+1])
            for lpi in lp:
                ostr = '{:d}-{:d}-{:d}-{:d}'.format(o.atom,o.n,o.l,lpi)
                if lpi>=0:
                    orbital_funcs[ostr] = gen_const(rad_dict['rad_args'][ostr])
                else:
                    orbital_funcs[ostr] = gen_const(0.0)
    
    return orbital_funcs
        
    
    
def gen_orb_labels(basis):
    
    '''
    Simple utility function for generating a dictionary of 
    atom-n-l:[Z, orbital label] pairs, to establish which radial integrals
    need be computed.
    
    *args*:   

        - **basis**: list of orbitals in basis
    
    
    *return*:

        - **orbitals**: dictionary of radial integral pairs
        
    ***    
    '''
    orbitals = {}
    for o in basis:
        o_str = '{:d}-{:d}-{:d}'.format(o.atom,o.n,o.l)
        if o_str not in orbitals.keys():
            orbitals[o_str] = [o.Z,o.label]
    return orbitals
    

def radint_calc(k_norm,orbital_funcs,phase_shifts=None):
    
    '''
    Compute dictionary of radial integrals evaluated at a single |k| value
    for the whole basis. Will avoid redundant integrations by checking for
    the presence of an identical dictionary key. The integration is done
    as a simple adaptive integration algorithm, defined in the 
    *adaptive_int* library.
    
    *args*: 

        - **k_norm**: float, length of the k-vector 
        (as an argument for the spherical Bessel Function)
        
        - **orbital_funcs**: dictionary, radial wavefunction executables
        
    *kwargs*:

        - **phase_shifts**: dictionary of phase shifts, to convey final state scattering
        
    *returns*:

        - **Bdic**: dictionary, key value pairs in form -- 'ATOM-N-L':*Bval*
        
    ***
    '''
    
    Bdic = {}
    
    for o in orbital_funcs:
        
            
        l = int(o.split('-')[2])
        L=[x for x in [l-1,l+1]]
        for lp in L:
            Blabel=o+'-'+str(lp)
            if phase_shifts is None:
                phase_factor = 1.0
            else:
                phase_factor = phase_shifts[Blabel]
            if Blabel not in Bdic.keys():
                if lp<0:
                    tmp_B = 0.0
                else:
                
                    integrand = adint.general_Bnl_integrand(orbital_funcs[o],k_norm,lp)
                    b = find_cutoff(integrand)
                    a = 0.001
                        
                    tol = 10.0**-10
                        
                    tmp_B = phase_factor*adint.integrate(integrand,a,b,tol)
                       
                Bdic[Blabel]=tmp_B
    return Bdic

def fill_radint_dic(Eb,orbital_funcs,hv,W=0.0,phase_shifts=None,fixed=False):
    
    '''
    Function for computing dictionary of radial integrals. 
    Can pass either an array of binding energies or a single binding
    energy as a float. In either case, returns a dictionary however 
    the difference being that the key value pairs will have a value
    which is itself either a float, or an interpolation mesh over 
    the range of the binding energy array. The output can then be 
    used by either writing **Bdic['key']** or
    **Bdic['key']**(valid float between endpoints of input array)
    
    *args*:   

        - **Eb**: float or tuple indicating the extremal energies
       
        - **orbital_funcs**: dictionary of executable orbital radial wavefunctions
       
        - **fixed**: bool, if True, constant radial integral for each scattering
        channel available: then the orbital_funcs dictionary already
        has the radial integral evaluated
      
        - **hv**: float, photon energy of incident light.
        
    *kwargs*:

        - **W**: float, work function
        
        - **phase_shifts**: dictionary for final state phase shifts, as an optional
        extension beyond pure- free electron final states. For now, float type.
        
    *return*:

        - **Brad**: dictionary of executable interpolation grids

    ***
    '''
    if fixed:
        Brad = {o:orbital_funcs[o] for o in orbital_funcs}
    else:

        if type(Eb)==float:
            kval = np.sqrt(2.0*me/hb**2*((hv-W)+Eb)*q)*A 
            return (radint_calc(kval,orbital_funcs) if ((hv-W)+Eb)>=0 else 0)
        elif type(Eb)==tuple:
            Brad_es=np.linspace(Eb[0],Eb[-1],5)
            BD_coarse={}
            for en in Brad_es:
                k_coarse = np.sqrt(2.0*me/hb**2*((hv-W)+en)*q)*A #calculate full 3-D k vector at this surface k-point given the incident radiation wavelength, and the energy eigenvalue, note binding energy follows opposite sign convention
                tmp_Bdic = (radint_calc(k_coarse,orbital_funcs,phase_shifts) if ((hv-W)+en)>=0 else {})
                for b in tmp_Bdic:
                    try:
                        BD_coarse[b].append(tmp_Bdic[b])
                    except KeyError:
                        BD_coarse[b] = [tmp_Bdic[b]]
    
            Brad = {}     
            for b in BD_coarse:
                f = interp1d(Brad_es,BD_coarse[b],kind='cubic')
                Brad[b] = f
    if type(Eb)!=float and type(Eb)!=tuple and type(Eb)!=list and type(Eb)!=np.ndarray:
        print('RADIAL INTEGRAL ERROR: Enter a valid energy (float) or range of energies (tuple of 2 floats indicating endpoints). Returning None')
        return None
    else:
        return Brad

def find_cutoff(func):
    
    '''
    
    Find a suitable cutoff lengthscale for the radial integration:
    Evaluate the function over a range of 20 Angstrom, with reasonable
    detail (dr = 0.02 A). Find the maximum in this range. The cutoff
    tolerance is set to 1/1e4 of the maximum value. Since this 'max'
    is actually a lower bound on the true maximum, this will only give
    us a more strict cutoff tolerance than is absolutely possible. With
    this point found, we then find all points which are within the
    tolerance of zero. The frequency of these points is then found. When
    the frequency is constant and 1 for all subsequent points, we have
    found the point of convergence. If the 'point of convergence' is the
    last point in the array, the radial wavefunction really isn't suitably
    localized and the user should not proceed without giving more
    consideration to the application of the LCAO approximation to such
    a function.
    
    *args*: 

        - **func**: the integrand executable
    
    *return*:

        - float, cutoff distance for integration
        
    ***    
    '''
    
    r_c = np.linspace(0,25,1000)[1:]
    func_c = func(r_c)
    f_max = abs(func_c).max()

    ftol = f_max/1e4
    sub_tol_inds = np.where(abs(func_c)<ftol)[0]
    d_inds = [sub_tol_inds[i+1]-sub_tol_inds[i] for i in range(len(sub_tol_inds[:-1]))]
    for i in range(len(d_inds)):
        if np.sum(d_inds[i:]) == len(d_inds[i:]):
            cutoff = i
            break
#
    if cutoff==len(d_inds):
        print('RADIAL WAVEFUNCTION ERROR: WAVE FUNCTION IS NOT WELL CONVERGED OVER 25 ANGSTROM. EXITING')
        f_max_ind = np.where(abs(func_c)==f_max)[0][0]
        sub_tol_pts = r_c[sub_tol_inds]
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(r_c,np.real(func_c))
        ax.plot(r_c,np.imag(func_c))
        ax.vlines(r_c[f_max_ind],-f_max*1.1,f_max*1.1,colors='r')
        for ii in sub_tol_pts:
            ax.vlines(ii,-f_max*1.1,f_max*1.1,colors='grey')
        ax.vlines(r_c[sub_tol_inds[cutoff]],-f_max*1.1,f_max*1.1,colors='green')

        return None
    else:
        return r_c[sub_tol_inds[cutoff]]
    
    
    
def gen_const(val):
    
    '''
    
    Create executable function returning a constant value
    
    *args*:  

        - **val**: constant value to return when executable function
    
    *return*:   
       
        - lambda function with constant value
    
    ***
    '''
    
    return lambda x: val
    