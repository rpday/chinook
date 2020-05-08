#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 22:31:45 2019

@author: ryanday
"""

import numpy as np
import graphene_backend


## If you're new to python, it's good practice to enclose any part of your script you want
## to run explicitly when the script is executed in an 'if __name__ == "__main__":'
## block. This code block will only be run if you run the script directly,
## and will not be run if you import this script for example. 
##It's a great place to put tests for example.

if __name__ == "__main__":
    
    
    ## We start by constructing the full sp3 model
    
    TB_full,kpath = graphene_backend.construct_tightbinding(pzonly=False)

    ## We can then diagonalize and plot the bandstructure
    TB_full.solve_H()
    TB_full.plotting()
    
    
    ## Follow this with calculation of fat-bands
    ## Can use for example TB.print_basis_summary() to see that the orbital basis
    ## is ordered as C2s = [0,4], C2px = [1,5], etc.
    
    ##TODO, create a list of orbitals for use in fatbands. For example,
    #### if I just want the C2s and the A-site px orbital projections, 
    ####projections would be [[0,4],[1]] 
    #### Use TB_full.print_basis_summary() to get a summary of orbital basis
    
#    projections = [] ##FILL IN THIS LIST with C2s, C2px, C2py, C2pz orbitals
#    projections = [[0,4],[1,5]]
#    graphene_backend.do_fatbands(TB_full,projections)
    ####
    
    ## From output, see that the pz are really the only required orbitals.
    
    ###
    TB_pz,kpath = graphene_backend.construct_tightbinding(pzonly=True)
    TB_pz.plotting()
    ###

    ### In the backend file, I have included a function for plotting the
    ### orbital wavefunction at a designated k-point. To for example plot
    ### the lower-energy state near K, I can run
#    graphene_backend.plot_wavefunction(TB=TB_pz,band_index=0,k_index=190)
    
    ##I've specified 190 as there are 200 - points between each high-symmetry point
    ## in our k-path.
    ##TODO plot the wavefunction on the upper state at the same k-point.
    ##Do the same for another point in momentum space.

    
    ### Now on to ARPES. The Brillouin zone is pretty big, so let's focus
    ##on the K-point. You can use print(TB_pz.Kobj.pts) to find the coordinates
    ## of the K-point
    
    ##TODO fill in with proper coordinates
    Kpt = np.array([1.702,0.0,0.0])
    
    
    experiment = graphene_backend.setup_arpes(TB_pz,Kpt=Kpt)
    
    ## And now we can actually plot the spectrum. 
    
#    Imap,Imap_resolution,axes = experiment.spectral(slice_select=('w',0))
    
    ##TODO, run spectral over a few other points to explore the region.


    ##In graphene, it's interesting to consider the presense of a gap term.
    ##I've written a helper function to add first a Semenoff mass. 

    ##TODO, create a new tight-binding model, using only the pz-basis
#    TB_sem,kpath = 
    ##With the model you defined here, now add a Semenoff mass:
#    mass = 0.0
#    TB_pz = []
    
    
    
#    graphene_backend.semenoff_mass(TB_sem,mass)
    

    ##TODO now solve the Hamiltonian for the TB_sem model, and plot its dispersion


    ##TODO plot the eigenfunction for the same states you did in the model
    ##without the Semenoff mass. What has changed?

    ##TODO set up an ARPES calculation over the same region as before.
    
    ##TODO Use the graphene_backend.haldane_mass(TB_haldane,mass) function to 
    ## generate a model with a Haldane mass
    
    
    ## TODO combine both Semenoff and Haldane masses in a single model
    
    ##TODO plot the dispersion over -K -> Gamma -> K to see inequivalence 
    ## of the two K-points in presence of inversion and time reversal symmetry 
    ## Hint, edit the *points* argument of momentum_args in
    ## graphene_backend.construct_tightbinding(), or build a new K-object using
    ## a similar dictionary as *momentum_args*

