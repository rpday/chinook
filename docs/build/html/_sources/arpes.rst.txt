.. toctree::
   :maxdepth: 2


ARPES Simulation
****************
In addition to the core ARPES_lib library, several other scripts in the module are written with the express purpose of facilitating calculation of the ARPES intensity. All relevant  docs are included below.


adaptive_int: 
=============

.. automodule:: chinook.adaptive_int
   :members:

ARPES_lib:
==========

.. automodule:: chinook.ARPES_lib
   :members:

intensity_map:
==============

.. automodule:: chinook.intensity_map
   :members:

matplotlib_plotter:
===================
A built-in data-explorer is included in chinook, built using matplotlib to ensure cross platform stability. The figure below shows an example screen capture for a calculation on :math:`Sr_2IrO_4`. The user has the ability to scan through the momentum and energy axes of the dataset, and the cursor can be used actively to select momentum- and energy- distribution curves in the side and lower panels. A scatterplot of the bare dispersion, as computed from the Hamiltonian diagonalization is plotted overtop the intensity map.

.. image:: images/matplotlib_plotter.png
   :width: 600

.. automodule:: chinook.matplotlib_plotter
   :members:

radint_lib: 
==========

.. automodule:: chinook.radint_lib
   :members:

tilt:
=====

.. automodule:: chinook.tilt
   :members:
