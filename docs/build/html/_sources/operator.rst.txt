.. toctree::
   :maxdepth: 2


Model Diagnostics
*****************

chinook is designed with the capacity to do fairly extensive characterization of the tight-binding model being used. These scripts contain useful tools for understanding the model in more detail

Density of States
=================

Several different approaches can be taken to the execution of density of states. One can follow Blöchl_ , partitioning the Brillouin zone into a cubic mesh, with each cube further divided into a group of identical tetrahedra. This approach provides the ability to perform matrix diagonalization over a fairly small number of k-points, as one interpolates within the tetrahedra to construct the full density of states. This is executed using *dos_tetra* defined below.

.. _Blöchl: https://doi.org/10.1103/PhysRevB.49.16223

Alternatively, one can perform diagonalization explicitly only at the nodes of a mesh defined over the Brillouin zone, and apply some broadened peak at the eigenvalues of the Hamiltonian at each point. Generally, for reasonably narrow Gaussian broadening this requires a fairly dense k-mesh. In practice, the large supercells where diagonalization becomes costly are also associated with much smaller Brillouin zones, allowing for a smaller k-space sampling. We find this method to perform better in most cases. This is executed using the *dos_broad* function defined below. 

Related tools are also available to find the Fermi level, given a specified electronic occupation (*dos.find_EF* using gaussian, and *dos.EF_find* using tetrahedra). 

.. automodule:: dos
   :members:

Fermi Surface
=============

We use a modified version of the method of `marching tetrahedra <https://doi.org/10.1088/1367-2630/16/6/063014>`_ to find the Fermi surface within the reciprocal lattice. The standard definition of the reciprocal space mesh runs over the primitive parallelepiped defined by the recriprocal lattice vectors. Shifts to the lattice origin are provided as an option. The loci of the Fermi surface are found for each tetrahedra, and used to assemble a continuous set of triangular patches which ultimately construct the Fermi surface for each band which crosses the Fermi level.


.. automodule:: FS_tetra
   :members:

Operators
=========

A number of tools are included for characterizing the expectation values of observables, as projected onto the eigenstates of the model Hamiltonian. The main function here is *O_path*, which will compute the expectation value of an indicated operator (represented by the user as a Hermitian numpy array of complex float with the same dimensions as the orbital basis). The resulting values are then displayed in the form of a colourmap applied to the bandstructure calculation along the desired path in momentum space. Several common operators are defined to facilitate these calculations, such as spin :math:`\hat S_i` and :math:`\left<\vec{L}\cdot\vec{S}\right>` . In addition, *fatbs* uses *O_path* to produce a "fat bands" type spaghetti plot, where the colourscale reflects the orbital projection onto the bandstructure. 

.. automodule:: operator_library
   :members:

Orbital Visualization
=====================

.. automodule:: orbital_plotting
   :members:

tetrahedra
===========

.. automodule:: tetrahedra
   :members: