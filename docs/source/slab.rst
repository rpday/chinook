.. toctree::
   :maxdepth: 2


Slab Calculation
****************

chinook facilitates the generation of slab-models based around bulk Hamiltonians defined
by the user. This functionality is in beta-testing mode for version 1.0. so please proceed with caution, and contact the developers if you have any concerns.

The setup for a slab-type calculation proceeds similarly to that for a bulk model. The call to build a tight-binding model passes an additional argument,
::
	TB = chinook.build_lib.gen_TB(basis_dict,hamiltonian_dict,K_object,slab_dict)

The *slab_dict* formats as follows.
::
	slab_dict = { 'avec':avec,
	      'miller':np.array([0,0,1]),
	      'thick':30,
	      'vac':20,
	      'termination':(0,0),
	      'fine':(0,0)}

Here one passes the lattice vectors *avec*, along with the desired Miller index of the surface termination. A desired thickness of both the slab and the vacuum buffer are also required. These are in units of Angstrom. This defines a lower bound: to find a suitable slab which satisfies the desired atomic termination, the actual slab can often be larger. The *vac* should be at very least longer than the farthest hopping vector considered in the model. This doesn't add any computational overhead *after* the definition of the slab model is established, so a large value is not a serious bottleneck in subsequent calculations. 

The *termination* tuple designates which inequivalent atoms in the basis one wants to have
on the top and bottom surface. If a mixed-character surface (e.g. TiO\ :sub:`2`\  plane) is desired, either species can be selected. Finally the *fine* tuple allows user to adjust the position of the termination, which can be necessary in the event of incorrect software selection of surface. This can occur commonly in layered materials, for example below for a AB\ :sub:`2`\  material: while a real material will generally terminate at the bottom of the van-der Waals layer, either instance of atom 0 can satisfy the *slab_dict* input. This is indicated in the figure:

.. image:: images/surface_selection.png
   :width: 600

Fine allows the user to request a shift of designated thickness to force the program to select the proper surface termination. As *slab* is in beta development, some diagnostics of the generated slab should be conducted by the user before proceeding with more complex calculations.

.. WARNING::
	Slab generation will rotate the global coordinate frame to place the surface normal along the :math:`\hat z` direction, and one of the in-plane vectors along the :math:`\hat x` direction. This may lead to unanticipated redirection of the high-symmetry points.

Slab Library
============

.. automodule:: slab
   :members:

Surface Vector
==============

.. automodule:: surface_vector
   :members:

.. automodule:: v3find
   :members:
