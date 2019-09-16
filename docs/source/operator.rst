.. toctree::
   :maxdepth: 2


Model Diagnostics
*****************

chinook is designed with the capacity to do fairly extensive characterization of the tight-binding model being used. These scripts contain useful tools for understanding the model in more detail

Density of States
=================

.. automodule:: dos
   :members:

Fermi Surface
=============

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