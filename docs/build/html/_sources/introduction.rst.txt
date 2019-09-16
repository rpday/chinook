.. toctree::
   :maxdepth: 2

Introduction
************

Installation
============

The entire source code for *chinook* is available as a github repository, which can be accessed
by visiting our repo_. 

.. _repo: https://github.com/rpday/chinook

For most convenient access, *chinook* is registered and available for download via *pip*. Open a command line shell and execute
::
  pip install chinook

We have avoided use of unconventional packages to run *chinook*, and the only required packages are numpy_, matplotlib_, and scipy_. Some high performance operation is available if psutil_ is also available.

.. _numpy: https://numpy.org

.. _matplotlib: https://matplotlib.org

.. _scipy: https://www.scipy.org/scipylib/index.html

.. _psutil: https://psutil.readthedocs.io/en/latest/

Getting Started
===============

At the core of any chinook calculation is the tight-binding model. The requisite information to construct such a model is an orbital basis and a Hamiltonian which effectively describes the Hilbert space spanned by this orbital basis.

In order to construct the essential objects in chinook, we make use of Python dictionaries, a hash-table representation which allows for an efficient, but more importantly human-readable, input format. If you open the template.py file in your chinook distribution, you will find the basic input for starting a calculation.

Lattice
=======

The lattice is defined as a 3x3 array called avec, and is defined in units of inverse Angstrom. The default provided in template.py is a unit-length simple cubic lattice. As an example here, we construct the unit cell of GaAs,
::
    avec = np.array([2.8162,2.8162,0],[0,2.8162,2.8162],[2.8162,0,2.8162]])


Basis
=====

The basis in chinook is defined as a list of orbital objects.
::
    basis_args = {'atoms':[0,1],
    'Z':{0:31,1:33},
    'pos':[np.array([0,0,0]),np.array([1.4081,1.4081,1.4081])],
    'orbs':[['40','41x','41y','41z'],['40','41x','41y','41z']]}

We see from this that the distinct atomic species (Ga and As) are indexed as 0 and 1 in the 'atoms' argument. Their atomic numbers Z are then given below in a dictionary format, with key-value pairs corresponding to the atom index, and the atomic number. The atom positions can then be given, in units of Angstrom. These are entered as numpy arrays, within a list which orders in the same way as atoms does. Finally, we define the orbitals at each of these basis sites which are to be included in our model. In this case, we include both s and all *px*, *py*, and *pz* orbitals for both Ga and As. Orbitals are labelled by the principal quantum number n, orbital angular momentum l and then a standard label. Details can be found in *chinook.orbital.py*. Note that each atom gets its own list of orbitals, rather than putting them all together in a single list. Finally, there are additional options, including rotations and spinful spinor bases which can be implemented. More details also in *chinook.orbital.py*.

Hamiltonian
===========
The Hamiltonian has a fair amount of flexibility. For further details, it is recommended to refer to the documentation under *chinook.TB_lib.py*. In the template, we are assuming that the user wishes to enter a Slater-Koster type Hamiltonian. In this case, the hopping terms need to be defined. We can use the parameters given by, for example Harrison or Cohen, which gives us a dictionary which looks like so:
::
    V_SK = {'040':-6.55,'041':0.48,'140':-8.84,'141':1.48,
            '004400S':-2.71,'004401S':-1.42,'004410S':2.71,
            '004411S':2.65,'004411P':-0.64,
            '114400S':-0.24,'114401S':0.01,
            '114411S':0.47,'114411P':-0.10}

In the above, we begin by defining the on-site energies, where the key-strings are formatted as 'atom-n-l', and the values are the energies in eV. Second, we define the various Slater-Koster type hoppings, such as Vsss, Vsps, Vpps, Vppp. These have key-strings as 'atom1-atom2-n1-n2-l1-l2-x'. With such a dictionary defined, the user can combine this with the other pertinent descriptors for the hamiltonian arguments, as seen in the template:
::
    hamiltonian_args = {'type':'SK',
                        'V':V_SK,
                        'avec':avec,
                        'cutoff':2.5,
                        'renorm':1.0,
                        'offset':0.0,
                        'tol':1e-4}

In addition to denoting that we will use a Slater-Koster type Hamiltonian, we pass the dictionary of potentials 'V'. In addition to the lattice vectors, we finally define a few numeric values, corresponding to the cutoff hopping lengthscale (in Angstrom), any overall renormalization factor for the bandstructure, an offset for the Fermi level, and a tolerance, or minimal Hamiltonian matrix element size we wish to consider--any smaller terms will be neglected.

Tight-Binding Model
===================

With these input arguments defined we can now actually build our tight-binding model and begin to test it. To do so, we first define the list of orbitals for our model:
::
    basis = chinook.build_lib.gen_basis(basis_args)

This is then used with the `hamiltonian_args` defined above to build the tight-binding model:
::
    TB = chinook.build_lib.gen_TB(basis,hamiltonian_args)

We now have a functioning tight binding model. However, to actually perform any work, we need to know where in momentum space we are interested in diagonalizing over. This is done using a similar argument structure as before.

Momentum
========
The momentum arguments get passed as a dictionary:
::
    L,G,X = np.array([0.5,0.5,0.5]),np.array([0.0,0.0,0.0]),np.array([0.5,0.5,0.0])
    momentum_args = {'type':'F',
                     'avec':avec,
                     'grain':100,
                     'pts':[L,G,X],
                     'labels':['L','$\\Gamma$','X']}

Here we are using *F* or fractional units for momentum space (as opposed to *A* absolute units of inverse Angstrom) to define our k-path. This requires also that we pass then the unit cell vectors. The *grain* sets the number of points between each high-symmetry point we want to evaluate at. The endpoints of interest are passed similar to what we did with the basis positions, as a list of numpy arrays, which I have pre-defined for tidier code. Finally, we have an option to provide labels for when we go ultimately to plot our bandstructure over this k-path. I can now set the k-path for my tight-binding model:
::
    TB.Kobj = chinook.build_lib.gen_K(momentum_args)

And then diagonalize and plot the band structure.
::
    TB.solve_H()
    TB.plotting()


ARPES Calculation
=================
Presumably, we're all here for the ARPES calculations. Once you have a tight-binding model you're happy with, you can proceed to initialize and execute and ARPES experiment. We do this with the following input
::
    arpes_args = {'cube':{'X':[-0.5,0.5,100],
                          'Y':[-0.5,0.5,100],
                          'kz':0.0,
                          'E':[-3,0.1,1000]},
                  'hv':21.2,
                  'SE':['constant',0.001],
                  'T':4.2,
                  'pol':np.array([1,0,0]),
                  'resolution':{'E':0.01,'k':0.01}}

This is sort of the most basic set of arguments we can define for an ARPES experiment, leaving most others as default. We have defined a *cube* of momentum and energy over which we are interested in evaluating the photoemission intensity. We set both the endpoints and grain for the momentum in-plane as well as the energy, given here as binding energy relative to the zero of our tight-binding model, which will be the Fermi level. A fixed out-of-plane momentum is chosen, and defined as *kz*. Along with this *cube*, we fix the photon energy for the experiment. With these two sets of parameters defined, the matrix elements can be calculated. As a result, all other arguments given here can be updated after evaluating the matrix elements, such that different parameter choices and their influence on the ARPES intensity can be surveyed very quickly and with little computational overhead. The first of these is the self-energy. Here, we the self-energy (i.e. *SE*) is taken to be purely imaginary, giving only the width as a constant 1 meV width for the peaks. Various other approximations are available, to which you are referred to the documentation of *chinook.ARPES_lib.py*, including Kramers-Kronig related Real and Imaginary parts of the self-energy. In addition, the Fermi function is evaluated here at 4.2 K to suppress intensity from unoccupied states. A polarization and both energy and momentum resolutions are also included. The units for the latter are in terms of eV and inverse Angstrom respectively, and are evaluated at FWHM. These parameters, along with our tight-binding model can then be used to seed an experiment
::
    experiment = chinook.arpes_lib.experiment(TB,arpes_args)
    experiment.datacube()
    Imap = experiment.spectral()
                  
In these three lines, we initialize the experiment, evaluate the matrix elements, and generate an ARPES intensity map. Please refer to the documentation for notes on further parameters available for these calculations, and methods for updating parameters following execution of *experiment.datacube()*.		            


