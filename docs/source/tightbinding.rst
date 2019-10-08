.. _tight binding: 

.. toctree::
   :maxdepth: 2


Tight Binding
*************

The user tight-binding model contains all relevant information regarding the orbital basis, the model Hamiltonian (in addition to eigenvalues/eigenvectors), as well as the momentum domain of interest. In addition, the tight-binding class contains all relevant methods required to extract this information.

The user has a reasonable amount of flexibility in the format which they would use to generate the model Hamiltonian. This is intended to accommodate various programs which can be used to generate a tight-binding Hamiltonian (for example Wannier90), as well as the different presentations used in publications (Slater-Koster :math:`V_{lmk}` , :math:`t_{ij}` ), as well as alternative Hamiltonian types, such as low-energy effective models which do not adhere to the full translational symmetry required by Bloch's theorem. These latter models do however provide a highly useful and physically transparent parameterization of the system Hamiltonian for narrow regions of momentum space near high-symmetry points. For these reasons, there are 4 general categories of Hamiltonian inputs we accept. The first three are described as: *Slater-Koster*, *list*, and *text* input. The last is described generically as *executable*.

While in principal a tight-binding Hamiltonian can be passed in the acceptable form for any of the above, the last option also supports these low-energy theories described above. 

Slater-Koster Models:
=====================

In their 1954 PRB paper, Slater and Koster presented a fairly simple framework for defining the tight-binding model associated with hoppings between localized atomic-like orbitals on neighbouring lattice sites. To define the overlap for a pair of basis states with orbital angular momentum :math:`l_1` and :math:`l_2` for an arbitrary lattice geometry, we require only :math:`(min( l_1, l_2 ) + 1)` parameters. For example, for :math:`l_1=l_2=1` we define :math:`V_{pp\sigma},\ V_{pp\pi}`. Intuitively, these parameters correspond to overlap integrals between the two orbitals when the lobes of the 'p' orbitals are aligned parallel to the connecting vector ( :math:`\sigma`) and aligned perpendicular to the connecting vector ( :math:`\pi` ). One can often use the frequently published table of Slater-Koster parameters to then define general rules for how these two parameters should be combined for a specific lattice geometry to arrive at the hopping amplitude between each pair of lattice sites. 

This table is however restrictive as it provides rules for only hoppings between non-distorted cubic harmonics. One can alternatively take advantage of the representation of the orbital states in the basis of spherical harmonics :math:`Y_{l}^{m}(\Omega)` to rotate an arbitrary pair of basis states into a common representation, and then rotate the frame of reference to align the bond-direction with a designated quantization axis: here the :math:`\hat z` vector. A diagonal Hamiltonian matrix filled with the associated :math:`V_{l_1l_2\gamma}` can then be applied. The rotation and basis transformation can be undone, ultimately producing a matrix of Hamiltonian parameters for the entire orbital shell along the designated bond vector. Mathematically, the procedure is represented by

.. math::
	\left<\psi|H|\phi\right> = \left<\psi| U^{\dagger} R^{-1}(\theta,\phi) V_{SK} R(\theta,\phi) U |\phi\right>

This formalism then allows for fast and user-friendly generation of a Hamiltonian over an arbitrary basis and geometry. Given only the :math:`V_{l_1,l_2,\gamma}` parameters and the lattice geometry, a full tight-binding Hamiltonian can be built. 

The Slater-Koster parameters are passed in the form of a dictionary, with the keys taking the form of :math:`a_1 a_2 n_1 n_2 l_1 l_2 \gamma`. For example, if I have two distinct atoms in my basis, where I have the Carbon 2p and Iron 3d :math:`e_g` orbitals, the dictionary may look like shown here
::
	V_SK = {'021':3.0,'13XY':0.0,'13ZR':-0.5,
	'002211S':1.0,'002211P':-0.5,
	'012312S':0.1,'012312P':0.6,
	'113322S':0.05,'113322P':0.7,'113322D':-0.03}

In the first line I have written the on-site terms. While in a simple case I may have a single onsite energy for each orbital shell, here I have distinguished the onsite energy of the :math:`d_{x^2-y^2}` and :math:`d_{3z^2-r^2}` states. In the second-fourth lines, I have written the p-p, p-d, and d-d hopping amplitudes. 

In many models, hopping beyond nearest neighbours are relevant, and will typically not have the same strength as the nearest neighbour terms. In these cases, we can pass a list of dictionaries to *H_dict*. For example
::
	H_dict = {'type':'SK',
			'V':[SK_dict1,SK_dict2],
			'cutoff':[3.5,5.0],
			...}

To include these next-nearest neighbour terms, I specify a list of hopping dictionaries, in addition to a list of cutoff distances, indicating the range of connecting vectors each of the indicated dictionaries should apply to. For this case, for connecting vectors where :math:`|R_{ij}|<3.5` we use *SK_dict1*, whereas for :math:`3.5\leq |R_{ij}|<5.0` we use *SK_dict2*. 

Hamiltonian Construction:
=========================
From these matrices, we proceed to build a list of *H_me* objects. This class is the standard representation of the Hamiltonian in *chinook*. Each instance of *H_me* carries integers labelling the associated basis indices. For example :math:`\left<\phi_3|H|\phi_7\right>` will be stored with *H_me*.i = 3, *H_me*.j = 7. We note here that the Hermiticity of the Hamiltonian allows one to explicitly define only the upper or lower diagonal of the Hamiltonian. Consequently, in *chinook*, we use only the *upper* diagonal, that is :math:`i\leq j`. 

In addition to basis indices, the *H_me* object carries the functional information of the matrix element in its *H_me.H* attribute. This will be a list  of connecting vectors and hopping strengths in the case of a standard tight-binding model, or a list of python executables otherwise. 

In standard format then one might have
::
	TB.mat_els[9].i = 3
	TB.mat_els[9].j = 3
	TB.mat_els[9].H = [[0.0,0.0,0.0,2.0],
			[1.5,0.0,0.0,3.7],
			[-1.5,0.0,0.0,3.7]]

This is the 10-th element in our model's list *TB.mat_els* of *H_me* objects. This *H_me* instance contains an on-site term of strength 2.0, and a cosine-like hopping term along the :math:`x` -axis of strength 3.7 eV. A closer consideration of the *H* attribute reveals the essential information. Each element of the list is a length-4 list of float, containing :math:`\vec{R_{ij}^n}` and :math:`t_{ij}^n`. Ultimately, the full matrix element will be expressed as

.. math::
	H_{ij}(\vec{k}) = \sum_{n} t_{ij}^n e^{i\vec{k}\cdot\vec{R_{ij}^n}}

For this reason, if one does not have access to a suitable Slater-Koster representation of the tight-binding model, we can bypass the methods described above, passing a list of prepared matrix elements directly to the generator of the *H_me* objects. To accommodate this, we then also accept Hamiltonian input in the form of *list*, where each element of the list is written as for example
::
	Hlist[10] = [3,3,1.5,0.0,0.0,3.7]
The elements then correspond to basis index 1, basis index 2, connecting vector x, y, and z components, and finally the :math:`t_{ij}` value. 
Similarly, a textfile of this form is also acceptable, with each line in the textfile being a comma-separated list of values
::
	3,3,1.5,0.0,0.0,3.7
	...

A list of *H_me* objects can then be built as above. 

Executable Hamiltonians:
========================

The *executable* type Hamiltonian will ultimately require a slightly modified form of the input, as we do not intend to express the matrix elements as a Fourier transform of the real-space Hamiltonian in the same way as above.

In this form, we should have

.. math::
	H_{ij}(\vec{k}) = \sum_{n} f_n(\vec{k})

This then requires a modified form of *H_me.H*. By contrast with the above, 
::
	TB.mat_els[9].H = [np.cos,my_hopping_func]

where *my_hopping_func* is a user-defined executable. It could be for example a 2nd order polynomial, or any other function of momentum. The essential point is that the executables *must* take an Nx3 array of float as their only argument. This allows for these Hamiltonians to fit seamlessly into the *chinook* framework.

.. WARNING::
	Automated tools for building these Hamiltonians are currently in the process of being built. Proceed with caution, and notify the developers of any unexpected performance.

For advanced users who would like to take advantage of this functionality now, 
::
	H_dict = {'type':'exec',
			'exec':exec_list,
			...}

where the item
::
	exec_list = [[(0,0),my_func1],
	[(0,1),my_func2],
	...]

The tuple of integers preceding the name of the executable corresponds to the basis indices. Here *my_func1* applies to the diagonal elements of my first basis member, and *my_func2* to the coupling of my first and second basis elements. As always, indexing in python is 0-based.

For the construction of the executable functions, we recommend the use of python's *lambda* functions. For example, to define a function which evaluates to :math:`\alpha k_x^2 + \beta k_y^2` , one may define
::
	def my_func_generator(alpha,beta):
		return lambda kvec: alpha*k[:,0]**2 + beta*k[:,1]**2

	#Define specific parameters for my executable functions
	a1 = 2.3
	b1 = 0.7

	a2 = 5.6
	b2 = 0.9
	
	#Define my executables
	my_func1 = my_func_generator(a1,b1)
	my_func2 = my_func_generator(a2,b2)

	#Populate a list of executables
	exec_list = [[(0,0),my_func1],
	[(0,1),my_func2]]

Please stay tuned for further developments which will facilitate more convenient construction of these Hamiltonians.

Below, we include the relevant documentation to the construction of tight-binding models.



Model Initialization
====================

.. automodule:: build_lib
   :members:


Hamiltonian Library
===================

.. automodule:: H_library
   :members:

Momentum Library
================

.. automodule:: klib
   :members:

Orbital Objects
===============

.. automodule:: orbital
   :members:

Slater Koster Library
=====================

.. automodule:: SlaterKoster
   :members:

Tight-Binding Library
=====================

.. automodule:: TB_lib
   :members:
