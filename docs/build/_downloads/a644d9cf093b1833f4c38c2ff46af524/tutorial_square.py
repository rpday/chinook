import numpy as np
import input_square
import chinook.operator_library as operators

if __name__ == "__main__":

	#generate tight-binding model
	TB = input_square.build_model()
	
	#solve model, plot bandstructure
	TB.solve_H()
	TB.plotting()

	#plot fat-bands, projected onto px, py, and pz orbitals
	px = operators.fatbs(proj=[0,3,6,9],TB=TB,Elims=(-5,5),degen=True) 
	py = operators.fatbs(proj=[1,4,7,10],TB=TB,Elims=(-5,5),degen=True)
	pz = operators.fatbs(proj=[2,5,8,11],TB=TB,Elims=(-5,5),degen=True)

	#plot <L.S> expectation value projected onto bandstructure
	LdS_matrix = operators.LSmat(TB)
	LdS = operators.O_path(LdS_matrix,TB,degen=True)

	#build ARPES experiment, for Fermi-surface exploration
	arpes_args,arpes_experiment_FS = input_square.build_ARPES(TB,input_square.fermi_surface_args)

	#execute, plot the Fermi surface, with initial polarization (1,0,-1), as defined by input_square.fermi_surface_args
	_ = arpes_experiment_FS.spectral(arpes_args,slice_select=('w',0.0))

	#change polarization to (0,1,0)
	arpes_args['pol'] = np.array([0,1,0])

	#repeat previous calculation, with new polarization
	_ = arpes_experiment_FS.spectral(arpes_args,slice_select=('w',0.0))

	#perform high-resolution calculation along a specified momentum direction
	arpes_args,arpes_experiment_KX = input_square.build_ARPES(TB,input_square.xmomentum_cut_args)

	_ = arpes_experiment_KX.spectral(slice_select=('y',0),plot_bands=True)

	#again, change polarization, to (0,1,0)
	arpes_args['pol'] = np.array([0,1,0])

	#expand the range of interest along kx
	arpes_args,arpes_experiment_wideX = input_square.build_ARPES(TB,input_square.widex_momentum_cut_args)

	#plot result
	_ = arpes_experiment_wideX.spectral(slice_select=('y',0),plot_bands=True)








