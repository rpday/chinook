# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 11:47:18 2018

@author: rday
"""
import sys
sys.path.append('/Users/ryanday/Documents/UBC/TB_ARPES/TB_ARPES-master 4/')

import numpy as np
import ubc_tbarpes.build_lib as build_lib
import ubc_tbarpes.NBL_TB as CuTB
import ubc_tbarpes.ARPES_lib as ARPES
import ubc_tbarpes.operator_library as ops



def SK_extend(SK,avec,alpha,d):
    num = 1
    num_symm = num*2+1
    pts= np.dot(np.array([[int(i/num_symm**2)-num,int(i/num_symm)%num_symm-num,i%num_symm-num] for i in range((num_symm)**3)]),avec)
    lens = (np.linalg.norm(pts,axis=1))
    len_unique = []
    for li in lens:
        if (li not in len_unique) and li>0:
            len_unique.append(li)
    len_unique = np.array(sorted(len_unique))
    scale_factors = np.exp(-alpha*(np.array(len_unique)-d))
    
    CUT = 1.1*len_unique
    print(CUT,scale_factors)
    SK_out = [{ci:scale_factors[i]*SK[ci] for ci in SK} for i in range(len(CUT))]
    
    return CUT,SK_out



if __name__=="__main__":
    a =  3.56
#    avec = np.array([[a/2,a/2,0],[0,a/2,a/2],[a/2,0,a/2]])
#    _,_,Nr = slib.gen_surface(avec,np.array([1,1,1])),TB.basis
    avec = np.around(2.51730014/2.51735729*np.array([[-1.2587, -0.7267,  2.0554],[ 1.2587, -0.7267,  2.0554],[-0.    ,  1.4534,  2.0554]]),4)
    
    CUT,REN,OFF,TOL=3.5,1,0,0.001
    G,X,W,L,K = np.zeros(3),np.array([0,0.5,0.5]),np.array([0.25,0.75,0.5]),np.array([0.5,0.5,0.5]),np.array([0.375,0.75,0.375])
	

#    eta = {'040':-20.14,'031':100.0,'032':-20.14,'004400S':-0.48,'004301S':1.84,'003311S':3.24,
#           '003311P':-0.81,'004302S':-3.16,'003312S':-2.95,'003312P':1.36,'003322S':-16.20,
#           '003322P':8.75,'003322D':0}
#    d = a/np.sqrt(2)
#    SK = SKlib.eta_dict_to_SK(eta,0.67,np.sqrt(0.5)*a)
#    CUT,SK = SK_extend(SK,avec,6./d,d)
    fnm = 'cu_params_pan.txt'
    tol = 0.0001
    SK,CUT = CuTB.gen_Cu_SK(avec,fnm,tol)
    OFF = CuTB.pair_pot(fnm,avec)/(-18)+1.55#-18#
#    SK = SK[0]
#    CUT = float(CUT[0])
    
#    SK = {"040":-2.408,"031":4.00,"032":-5.00,"004400S":-0.05652552534871882,"003410S":0.10235422439959821,"004302S":-0.036994370838375354,
#          "003311S": 0.21924434953910618,"003311P":0.0,"003312S":-0.053580360262257626,"003312P":0.013802597521248922,"003322S":-0.012755569410002986,
#          "003322P":0.0033741803350209204,"003322D":-0.0012785070616424799}
#    txtfile = 'C:/Users/rday/Documents/TB_ARPES/2018/TB_ARPES_2018/TB_ARPES-master/examples/Cu.txt'

    spin = {'bool':True,'soc':True,'lam':{0:0.1}}

    Bd = {'atoms':[0],
			'Z':{0:29},
			'orbs':[["40","31x","31y","31z","32xy","32xz","32yz","32ZR","32XY"]],
			'pos':[np.zeros(3)],
            'spin':spin}

    Kd = {'type':'F',
          'avec':avec,
			'pts':[-G,-X,-W,-L,-G,-K],
			'grain':100,
			'labels':['$\Gamma$','X','W','L','$\Gamma$','K']}


    Hd = {'type':'SK',
			'V':SK,
          'avec':avec,
			'cutoff':CUT,
			'renorm':REN,
			'offset':OFF,
			'tol':TOL,
			'spin':spin}
    
    slab_dict = {'avec':avec,
      'miller':np.array([1,1,1]),
      'thick':16,
      'vac':10,
      'fine':(0,0),
      'termination':(0,0)}
    
    Bd = build_lib.gen_basis(Bd)
    Kobj = build_lib.gen_K(Kd)
    TB = build_lib.gen_TB(Bd,Hd,Kobj)
    

    
    
    G,M,K=np.zeros(3),np.array([0.5,0.5,0.0]),np.array([1./3,2./3,0.0])
    G,M,K = np.zeros(3),np.array([1.24795,0.72052,0.0]),np.array([0.83197,1.44103,0.0])
    
    Kd['type']='A'
    Kd['avec']=TB.avec
    Kd['pts'] = [G,M,K,G]
    TB.Kobj = build_lib.gen_K(Kd)
    Hd['avec'] = TB.avec

    TB.solve_H()
    TB.plotting()
##    
#    theta =np.linspace(0,2*np.pi,200)
#    kvals = np.array([[0.1*np.cos(t),0.1*np.sin(t),0] for t in theta])
# 
#    
#    
#    TB.Kobj.kpts = kvals
#    TB.Kobj.kcut = theta
#    TB.Kobj.kcut_brk = [0,2*np.pi]
#    TB.Kobj.labels=['0','2pi']
#    TB.solve_H()
###    TB.plotting(-1.5,1.5)
####    
#    Svx = ops.S_vec(len(TB.basis),np.array([1,0,0]))
#    Svy = ops.S_vec(len(TB.basis),np.array([0,1,0]))
#    Sproj = ops.surface_proj(TB.basis,10)
#    Sxs = np.dot(Svx,Sproj)
#    Sys = np.dot(Svy,Sproj)
#    SX = ops.O_path(Sxs,TB,Kobj=TB.Kobj,vlims=(-0.25,0.25),Elims=(-1,-0.),degen=True)
#    SY = ops.O_path(Sys,TB,Kobj=TB.Kobj,vlims=(-0.25,0.25),Elims=(-1,-0.),degen=True)
#    
    
#    ARPES_dict={'cube':{'X':[-2,2,140],'Y':[-2,2,140],'kz':0.0,'E':[-0.75,0.05,120]},
#        'SE':[0.005],
#        'directory':'/Users/ryanday/Documents/UBC/TB_ARPES-082018/examples/FeSe',
#        'hv': 50,
#        'pol':np.array([0,1,0]),
#        'mfp':7.0,
#        'slab':True,
#        'resolution':{'E':0.005,'k':0.01},
#        'T':[True,10.0],
#        'W':4.0,
#        'angle':0,
#        'spin':None,
#        'slice':[False,-0.005]}
#    
#    ARPES_expmt = ARPES.experiment(TB,ARPES_dict)
#    ARPES_expmt.plot_gui(ARPES_dict)