#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 22:31:45 2019

@author: ryanday
"""

import numpy as np
import graphene_backend




if __name__ == "__main__":
    
    
    ## We start by constructing the full sp3 model
    
#    TB_full,kpath = graphene_backend.construct_tightbinding(pzonly=False)

    ## We can then diagonalize and plot the bandstructure
#    TB_full.solve_H()
#    TB_full.plotting()
    
    
    ## Follow this with calculation of fat-bands
    ## Can use for example TB.print_basis_summary() to see that the orbital basis
    ## is ordered as C2s = [0,4], C2px = [1,5], etc.
    
    ##TODO, create a list of orbitals for use in fatbands. For example,
    #### if I just want the C2s and the A-site px orbital projections, 
    ####projections would be [[0,4],[1]] 
    #### Use TB_full.print_basis_summary() to get a summary of orbital basis
    
#    projections = [] ##FILL IN THIS LIST with C2s, C2px, C2py, C2pz orbitals
    
#    graphene_backend.do_fatbands(TB_full,projections)
    ####
    
    ## From output, see that the pz are really the only required orbitals.
    
    ###
    TB_pz,kpath = graphene_backend.construct_tightbinding(pzonly=True)
    TB_pz.plotting()
    ###
    
    ### Now on to ARPES. The Brillouin zone is pretty big, so let's focus
    ##on the K-point. You can use print(TB_pz.Kobj.pts) to find the coordinates
    ## of the K-point
    
    ##TODO fill in with proper coordinates
    Kpt = np.array([1.702,0,0])
    
    
    experiment = graphene_backend.setup_arpes(TB_pz,Kpt=Kpt)
    
    ## And now we can actually plot the spectrum. 
    
    Imap,Imap_resolution,axes = experiment.spectral(slice_select=('w',0))
    
    ##TODO, run spectral over a few other points to explore the region.
    
    
    
    
    