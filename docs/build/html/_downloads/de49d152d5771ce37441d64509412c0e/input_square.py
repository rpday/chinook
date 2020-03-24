#Import the requisite modules for calculation
import numpy as np
import chinook.build_lib as build_lib
import chinook.ARPES_lib as arpes_lib


def build_model():

	'''
	Construct tight-binding model

	*return*:

		- **TB**: tight-binding model object, including Hamiltonian, lattice and orbital basis

	'''

	####LATTICE GEOMETRY####
	a,c = np.sqrt(0.5)*5.0,5.0
	avec = np.array([[a,a,0],
		[a,-a,0],
		[0,0,c]])

	####MOMENTUM PATH####
	kpoints = np.array([[0.5,0.5,0.0],[0.0,0.0,0.0],[0.5,-0.5,0.0]])
	labels = np.array(['$M_x$','$\Gamma$','$M_y$'])
	kdict = {'type':'F',
	        'avec':avec,
	        'pts':kpoints,
	        'grain':200,
	        'labels':labels}

	k_object = build_lib.gen_K(kdict)

	####ORBITAL BASIS DEFINITION####
	spin = {'bool':True,  #include spin-degree of freedom: double the orbital basis
	       'soc':True,    #include atomic spin-orbit coupling in the calculation of the Hamiltonian
	       'lam':{0:0.5}} #spin-orbit coupling strength in eV, require a value for each unique species in our basis

	Sb1 = np.array([0.0,0.0,0.0])
	Sb2 = np.array([a,0,0])

	basis = {'atoms':[0,0], #two equivalent atoms in the basis, both labelled as species #0
	        'Z':{0:51},     #We only have one atomic species, which is antimony #51 in the periodic table.
	        'orbs':[['51x','51y','51z'],['51x','51y','51z']], #each atom includes a full 5p basis in this model, written in n-l-xx format
	        'pos':[Sb1,Sb2], #positions of the atoms, in units of Angstrom
	        'spin':spin} #spin arguments.


	basis_object = build_lib.gen_basis(basis)

	####HAMILTONIAN DEFINITION####
	Ep = 0.7
	Vpps = 0.25
	Vppp = -1.0
	VSK = {'051':Ep,'005511S':Vpps,'005511P':Vppp}
	cutoff = 1.01*a

	######################################################################
	#####Option: nearest neighbour, and next nearest neighbour hopping  ##
	V1 = {'051':Ep,'005511S':Vpps,'005511P':Vppp}                     ##
	V2 = {'005511S':Vpps/a,'005511P':Vppp/a}                          ##
	VSK = [V1,V2]                                                     ##
	cutoff = [1.01*a,1.56*a]                                            ##
	######################################################################

	hamiltonian = {'type':'SK',     #Slater-Koster type Hamiltonian
	              'V':VSK,          #dictionary (or list of dictionaries) of onsite and hopping potentials
	               'avec':avec,     #lattice geometry
	              'cutoff':cutoff,  #cutoff length-scale for hopping
	              'renorm':1.0,     #renormalize bandwidth of Hamiltonian 
	               'offset':0.0,    #offset the Fermi level
	              'tol':1e-4,       #minimum amplitude for matrix element to be included in model.
	              'spin':spin}      #spin arguments, as defined above


	TB = build_lib.gen_TB(basis_object,hamiltonian,k_object)
	return TB

def build_ARPES(TB,arpes_args):
    
	'''
	Construct ARPES experiment, and calculate the associated matrix elements,
	based on the input experimental arguments and the associated tight-binding model.

	*args*:

		- **TB**: tight-binding model object

		- **arpes_args**: dictionary of experimental parameters

	*return*:

		- **arpes_args**: dictionary of experimental parameters

		- **arpes_experiment**: experiment object

	'''
    
	arpes_experiment = arpes_lib.experiment(TB,arpes_args)
	arpes_experiment.datacube()

	return arpes_args,arpes_experiment




fermi_surface_args = {'cube':{'X':[-0.628,0.628,300],'Y':[-0.628,0.628,300],'E':[-0.05,0.05,50],'kz':0.0}, #domain of interest
        'hv':100,                          #photon energy, in eV
         'T':10,                           #temperature, in K
        'pol':np.array([1,0,-1]),           #polarization vector
        'SE':['constant',0.02],            #self-energy, assume for now to be a constant 20 meV for simplicity
        'resolution':{'E':0.02,'k':0.02}}  #resolution

xmomentum_cut_args = {'cube': {'X':[-0.628,0.628,200],'Y':[0,0,1],'kz':0,'E':[-4.5,0.1,1000]},
        'hv':100,                          #photon energy, in eV
         'T':10,                           #temperature, in K
        'pol':np.array([1,0,-1]),           #polarization vector
        'SE':['constant',0.1],            #self-energy, assume for now to be a constant 20 meV for simplicity
        'resolution':{'E':0.02,'k':0.02}}  #resolution

widex_momentum_cut_args = {'cube': {'X':[-1.256,1.256,300],'Y':[0,0,1],'kz':0,'E':[-5,0.2,1000]},
        'hv':100,                          #photon energy, in eV
         'T':10,                           #temperature, in K
        'pol':np.array([1,0,-1]),           #polarization vector
        'SE':['constant',0.1],            #self-energy, assume for now to be a constant 20 meV for simplicity
        'resolution':{'E':0.02,'k':0.02}}  #resolution


