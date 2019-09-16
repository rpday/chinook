# -*- coding: utf-8 -*-

#Created on Tue Jun 13 17:22:12 2017

#@author: rday

#MIT License

#Copyright (c) 2018 Ryan Patrick Day

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

#This script handles calculation of the Radial Integrals:
#    $int(dr r^3 j_lp(kf*r) R_nl(r))$
#for the photoemission matrix element calculations.
#It loads the radial part of the wavefunction from electron_configs, which itself
#takes Z and orbital label as arguments to do so.
#The integration technique is based on a simple adaptive integrations algorithm
#which is a sort of divide and conquer approach. For each iteration, the interval
#is halved and the Riemann sum is computed. If the difference is below tolerance,
#either one or both of the halves will return. If not, if either or both return diff
#above tolerance, than the corresponding interval is again split and the cycle repeats
#until the total difference over the domain is satisfactorily below tolerance.
#In this way, the partition of the domain is non-regular, with regions of higher
#curvature having a finer partitioning than those with flat regions of the function.


import scipy.special as sc


def general_Bnl_integrand(func,kn,lp):
    
    '''
    Standard form of executable integrand in the e.r approximation of the matrix element
    
    *args*:

        - **func**: executable function of position (float), in units of Angstrom
        
        - **kn**: float, norm of the k vector (in inverse Angstrom)
        
        - **lp**: int, final state angular momentum quantum number
    
    *return*:

        - executable function of float (position)
    
    ***
    '''
    
    def lambda_gen():
        return lambda r: (-1.0j)**lp*r**3*sc.spherical_jn(lp,r*kn)*func(r)
    
    return lambda_gen()

def rect(func,a,b):
    
    '''
    
    Approximation to contribution of a finite domain to the integral, 
    evaluated as a rough rectangle
    
    *args*:

        - **func**: executable to evaluate
        
        - **a**: float, start of interval
        
        - **b**: float, end of interval
        
    *return*: 

        - **recsum**: (complex) float approximated area of the region under
        function between **a** and **b**
    
    ***
    '''
    mid = (a+b)/2.0
    jac = abs(b-a)/6.0
    recsum = jac*(func(a)+4*func(mid)+func(b))
    return recsum

def recursion(func,a,b,tol,currsum):
    '''
    
    Recursive integration algorithm--rect is used to approximate the integral
    under each half of the domain, with the domain further divided until
    result has converged
    
    *args*:

        - **func**: executable
        
        - **a**: float, start of interval
        
        - **b**: float, end of interval
        
        - **tol**: float, tolerance for convergence
        
        - **currsum**: (complex) float, current evaluation for the integral
        
    *return*: 

        - recursive call to the function if not converged, otherwise the result as complex (or real) float
    
    ***
    '''
    mid = (a+b)/2.0
    sum_left = rect(func,a,mid)
    sum_right = rect(func,mid,b)
    err = abs(sum_left+sum_right-currsum)
    if err<=15*tol:
        return sum_left+sum_right
    return recursion(func,a,mid,tol/2.,sum_left)+recursion(func,mid,b,tol/2.,sum_right)

def integrate(func,a,b,tol):
    '''
    Evaluate the integral of **func** over the domain covered by **a**, **b**. This 
    begins by seeding the evaluation with a maximally coarse approximation
    to the integral.
    
    *args*:

        - **func**: executable
        
        - **a**: float, start of interval
        
        - **b**: float, end of interval
        
        - **tol**: float, tolerance for convergence
       
    
    *return*:
    
        - **Q**: (complex) float, value of the integral
    
    ***
    '''

    Q = recursion(func,a,b,tol,rect(func,a,b))
    return Q
