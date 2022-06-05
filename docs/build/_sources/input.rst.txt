.. _input arguments: 

.. toctree::
   :maxdepth: 2

Input Arguments
***************

Most high-level functions in *chinook* operate on dictionaries as input format. This is done to support an efficient and high-density, yet largely human-readable input. Perhaps more importantly, this offers a high degree of flexibility in terms of required and optional keyword arguments. This comes however with a caveat that the full breadth of optional and required input arguments are not always conveniently accessible. This page then contains a comprehensive summary of the input dictionaries for a variety of functions.


Basis
=====

.. csv-table::
   :header: "Argument", "Required", "Datatype", "Example", "Notes"
   :widths: 20, 10, 50,60, 200
   :stub-columns: 1

   "atoms", "Yes", "list of int", "[0,0,1]",  "Unique elements are assigned distinct integers, consecutive"
   "Z", "Yes", "dict of int:int", "{0:77,1:8}", "Atomic number for each atom from *atoms* above"
   "orbs", "Yes", "| list of lists of
   | string", "| [['60'],
   | ['60'],
   | ['21x','21y']]", "| Elements define the orbital, one list for each atom,
   | elements define the orbital definition. Structure
   | of strings are: *principal quantum number n*,
   | *orbital angular momentum l*, 
   | orbital label (s: N/A, p: x,y,z d:xz,yz,xy,XY,ZR)"
   "pos", "Yes", "list of numpy arrays", "| [np.array([0.0,0.0,0.0]),
   | np.array([1.2,0.0,0.0]),
   | np.array([0.7,5.4,1.2])]", "| Positions of each atom in basis, each should be 
   | 3-float long, written in units of Angstrom"
   "spin", "No", "dictionary", "see below", "Optional, for including spin-information, see below"
   "slab", "No", "dictionary", "see below", "Optional, for generating a slab-superstructure, see below"
   "orient", "No", "list", "[np.pi/3,0,-np.pi/3]", "| One entry for each atom, indicating a
   | local rotation of the indicated atom, various formats accepted. 
   | For more details, c.f. **chinook.orbital.py**"

Spin
====

.. csv-table:: 
   :header: "Argument", "Required", "Datatype", "Example", "Notes"
   :widths: 20, 10, 50,60, 200
   :stub-columns: 1

   "bool", "Yes", "boolean", "True", "| Switch for activating spin-degree of freedom.
   | Doubles basis size."
   "soc", "No", "boolean", "True", "Switch for activating spin-orbit coupling."
   "lam","Yes*","dictionary","{0:0.15,1:0.0}", "Required if soc:True, atomic-SOC strength"
   "order","No", "char", "F", "Spin-ordering, 'F' for ferro, 'A' for antiferro, None otherwise"
   "dS", "No*", "float", "0.5","| Required if order if 'F' or 'A'. 
   | Energy splitting between up and down in eV"
   "p_up", "No*", "numpy array of 3 float", "np.array([0.0,0.0,0.0])", "| For AF, position of spin-up states,
   | spin-up shifted down by *dS* on these sites."  
   "p_dn", "No*", "numpy array of 3 float", "np.array([0.0,0.0,0.0])", "| For AF, position of spin-down states,
   | spin-down shifted down by *dS* on these sites."   


Hamiltonian
===========

.. csv-table:: 
   :header: "Argument", "Required", "Datatype", "Example", "Notes"
   :widths: 20, 10, 50,60, 200
   :stub-columns: 1

   "type", "Yes", "string", "SK", "| type of Hamiltonian. Any of
   | 'SK' for Slater-Koster, 'txt' for textfile, 
   | 'list' for list of float, 'exec' for executable"
   "cutoff", "Yes", "float or list of float", "[1.5,3.6]", "| Cutoff hopping distance.
   | Can be either a single float, or list of floats
   | if different Hamiltonian arguments apply for
   | different neighbour distances. Units of Angstroms."
   "renorm", "Yes", "float", "1.0", "overall bandwidth renormalization"
   "offset", "Yes", "float", "0.1", "overall chemical potential shift in eV"
   "tol", "Yes", "float", "1e-4", "Minimum strength term to include in eV"
   "avec","*No","numpy array of float","| np.array([[1,0,0],
   | [0,1,0],
   | [0,0,1])","Lattice vectors 3x3 float, required with SK."
   "spin", "No", "dictionary", "See above", "spin information, see above"
   "V", "No", "dictionary", "{'020':0.0,'002200S':0.5}", "| Slater Koster arguments, if applicable.
   | See Slater-Koster on Tight-Binding page."
   "list","No", "list", "[[0,0,0,0,0,5],...]"," List of Hamiltonian matrix elements,
   | if passing in list format."
   "filename", "No", "string", "my_hamiltonian.txt", "Path to Hamiltonian textfile."
   "exec","No", "list of methods","see Executable page","list of executable Python functions, see relevant page."

Momentum Path
=============

.. csv-table:: 
   :header: "Argument", "Required", "Datatype", "Example", "Notes"
   :widths: 20, 10, 50,60, 200
   :stub-columns: 1

   "type", "Yes", "char", "F", "| type of units, either fractional,
   | i.e. units of reciprocal lattice units
   | 'F', or absolute 'A' (i.e. 1/A)"
   "pts", "Yes", "list of numpy array","| [np.array([0,0,0]),
   | np.array([0.5,0.0.0])]", "| Endpoints for the momentum path.
   | An arbitrary number of points (>1) can be selected."
   "grain", "Yes", "int", "200", "| Number of points between
   | each endpoint in the list of 'pts'."
   "labels", "No", "list of str", "['\\Gamma', 'M', 'X']", "| Labels for each of 'pts', supports
   | use of MathTex"


Slab
====

.. csv-table:: 
   :header: "Argument", "Required", "Datatype", "Example", "Notes"
   :widths: 20, 10, 50,60, 200
   :stub-columns: 1

   "hkl", "Yes", "numpy array of int", "np.array([0,0,1])", "Miller index of surface"
   "cells", "Yes", "float", "100.0", "Minimum thickness of slab, in Angstroms"
   "buff", "Yes", "float", 10.0", "Minimum thickness of vacuum buffer layer."
   "term", "Yes", "tuple of two int", "(1,1)", "| Atomic identities of atoms
   | on top and bottom of surface, 
   | as defined in the basis arguments"
   "fine", "Yes", "tuple of two float", "(0.0,0.0)", "| Fine adjustments to surface
   | termination. See slab page for details."


ARPES Experiment
================

.. csv-table:: 
   :header: "Argument", "Required", "Datatype", "Example", "Notes"
   :widths: 20, 10, 50,60, 200
   :stub-columns: 1

   "cube", "Yes", "dictionary", "| {'X' : [-0.5,0.5,100],
   | 'Y' : [-1,1,200],
   | 'E' : [-2,0.1,1000],
   | 'kz' : 0.0}", "| Region of interest, X, and Y
   | are momentum range, E is energy. 
   | Fixed kz only. For a single slice,
   | can use e.g. 'X' : [0,0,1] .Can also
   | pass angles (rad) using 'Tx', 'Ty'
   | instead of 'X', 'Y'"
   "pol", "Yes", "numpy array of float", "np.array([1,0,0])", "Polarization vector"
   "hv", "Yes", "float", "21.2", "Photon energy, in eV"
   "T", "Yes", "float", "4.2", "| Temperature, for Fermi distribution. 
   | -1, or do not include 'T', to neglect
   | thermal distribution."
   "resolution", "Yes", "dictionary", "| {'E' : 0.01, 
   | 'k' : 0.005}", "| Energy, and momentum 
   | (FWHM) resolution, in eV, 1/A."
   "SE", "Yes", "list", "['constant', 0.05]", "| Self-Energy, 
   | see ARPES for further details"
   "angle", "No", "float", "1.59", "Azimuthal rotation, in radians."
   "W", "No", "float", "4.5", "Work function, in eV"
   "Vo", "No", "float", "11.5", "Inner potential in eV, for kz-dependence"
   "slab", "No", "bool", "True", "| Specify if using a slab geometry,
   | will impose mean-free path. This
   | also truncates the eigenvectors."
   "mfp", "No", "float", "10.0", "| Electron mean-free path, 
   | sets penetration depth for slab-calculations."
   "spin", "No", "list", "[1,np.array([0,0,1])]", "| Spin-ARPES, specify spin 
   | projection (+/- 1 for up/down) and axis
   | as numpy array of 3 float."
   "threads", "No", "int", "4", "Split calculation across multiple parallel threads."
   "rad_type", "No", "string", "See below", "Radial integral type, See below for details."
   "rad_args", "No", "string", "See below", "Radial integral arguments, see below."

Radial Integrals
================

.. csv-table:: 
   :header: "Argument", "Datatype", "Notes"
   :widths: 20, 60, 200
   :stub-columns: 1

   "rad_type", "string", "| Type of initial radial 
   | state wavefunctions:
   | 'slater' (default), 'hydrogenic', 'grid' (sample function on
   | mesh), 'exec' for user-defined executable 
   | function, or 'fixed', where integrals are over-ridden for
   | constant values."
   "rad_args",, "| if 'rad_type' is 'grid',
   | pass scaling of grid as a numpy array
   | of float. If 'fixed', pass dictionary
   | using 'a-n-l-lp' keys, float values
   | (e.g. '0-2-1-0' for 2p to s transition)."
   "phase_shifts","dictionary", "| Final state phase
   | shifts as first attempt at scattering
   | final states. Same input type as for 'fixed' above,
   | (e.g. '0-2-1-0' for 2p to s transition)."
  
