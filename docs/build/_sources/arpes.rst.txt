.. toctree::
   :maxdepth: 2


ARPES Simulation
****************
In addition to the core *ARPES_lib* library, several other scripts in the module are written with the express purpose of facilitating calculation of the ARPES intensity. All relevant  docs are included below.

In setting up an ARPES calculation, one requires an existing model system, i.e. instance of the :obj:`TB_lib.TB_model` class. This includes all relevant information regarding the orbital basis and the model Hamiltonian. In addition to this, a number of experimental parameters should be specified. Similar to the input for defining an orbital basis and a Hamiltonian, we use python *dictionaries* to specify these details. An example input is shown here.
::
	ARPES_dict = {'cube':{'X':[-1,1,200],'Y':[-0.5,0.5,100],'E':[-1,0.05,1000],'kz':0},
			'hv':21.2,
			'pol':np.array([1,0,0]),
			'T':4.2,
			'SE':['fixed',0.02],
			'resolution':{'dE':0.005,'dk':0.01}}

By passing this dictionary to the input statement
::
	experiment = ARPES_lib.experiment(TB,ARPES_dict)

We initialize an ARPES experiment which will diagonalize the Hamiltonian over a 200x100 point mesh in the region :math:`-1 \leq k_x \leq 1` and :math:`-0.5 \leq k_y\leq0.5` . For states in the range of energy :math:`-1\leq E\leq 0.05` eV of the Fermi level, we will explicitly compute the ARPES cross section when requested. The calculation will be done with photon energy for He-1 :math:`\alpha` at 21.2 eV, and a sample temperature of 4.2 K. This carries into the calculation in the form of a Fermi-Dirac distribution, which suppresses intensity from states with positive energies. The polarization of light is taken to be along the *x* direction, and is indicated by length-3 array associated with the keyword *pol*.

The *SE* key indicates the form of the self-energy to be imposed in evaluating the lineshape of the spectral features associated with each peak. Here we impose a fixed-linewidth of 20 meV on all states, to allow us to focus on the matrix elements alone, without further complications from energy-dependent lineshape broadening. As detailed in the :meth:`ARPES_lib.SE_gen` below, more sophisticated options are available.

Finally, resolution is passed as well, with arguments for both energy and momentum resolution, expressed as full-width half maximum values, in units of electron volts and inverse Angstrom. Many more non-default options are available, including sample rotations, spin-projection (for spin-ARPES) and radial-integrals. See the documentation for :obj:`ARPES_lib.experiment` for further details.

Once a calculation of the matrix elements is completed, one is interested in plotting the associated intensity map. There are several options for this. First, the intensity map must be built using :meth:`ARPES_lib.experiment.spectral`. Note that GUI tools do this automatically, without the user's input. The :meth:`ARPES_lib.experiment.spectral` method will apply the matrix element associated with each state, to its spectral function, and sum over all states to produce a complete dataset. It generates a raw and resolution broadened intensity map. The *slice_select* option allows for plotting specific cuts in energy or momentum. Once a full dataset has been generated, this can be passed to :meth:`ARPES_lib.experiment.plot_intensity_map` to quickly image a different cut. Alternatively, interactive GUI tools are available under :ref:`Matplotlib Plotter`, or if the user has Tkinter installed, :meth:`ARPES_lib.experiment.plot_gui()`. 

Numerical Integration 
=====================
For evaluation of radial integrals

.. math::
   B_{n,l}^{l'}(k) = (i)^{l'}\int dr R_{n,l}(r) r^3 j_{l'}(kr)

we use an adaptive integration algorithm which allows for precise and accurate evaluation of numeric integrals, regardless of local curvature. We do this by defining a partition of the integration domain which is recursively refined to sample regions of high curvature more densely. This is done until the integral converges to within a numerical tolerance. 


.. automodule:: adaptive_int
   :members:

   The user is given some opportunity to specify details of the evaluation of radial integrals :math:`B_{n,l}^{l'}(k)` used in the calculations. Specifications can be passed to the ARPES calculation through the *ARPES_dict* argument passed to the :obj:`ARPES_lib.experiment` object. 

.. automodule:: radint_lib
   :members:

ARPES Library
=============

.. automodule:: ARPES_lib
   :members:

Intensity Maps
==============

.. automodule:: intensity_map
   :members:

Matplotlib Plotter
==================
A built-in data-explorer is included in chinook, built using matplotlib to ensure cross platform stability. The figure below shows an example screen capture for a calculation on :math:`Sr_2IrO_4` . The user has the ability to scan through the momentum and energy axes of the dataset, and the cursor can be used actively to select momentum- and energy- distribution curves in the side and lower panels. A scatterplot of the bare dispersion, as computed from the Hamiltonian diagonalization is plotted overtop the intensity map.

.. image:: images/matplotlib_plotter.png
   :width: 600

.. automodule:: matplotlib_plotter
   :members:

Tilt Geometry
=============
One can account for rotation of the experimental geometry during acquisition of a particular dataset by performing an inverse rotation on the incoming light vector. Similarly, spin projections can also be rotated into the laboratory frame to reflect the effect of a misaligned or rotated sample in a spin-ARPES experiment.

.. automodule:: tilt
   :members:
