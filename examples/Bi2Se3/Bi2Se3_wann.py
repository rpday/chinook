import numpy as np
import sys
sys.path.append("/Users/ryanday/Documents/UBC/TB_ARPES/TB_ARPES-master 02102018/")
import ubc_tbarpes.build_lib as build_lib
import ubc_tbarpes.ARPES_lib as ARPES 


avec=np.array([[1.1960,-2.0715,9.5453],[1.1960,2.0715,9.5453],[-2.3920,0.0000,9.5453]])

spin_dict = {"bool":False,
	"soc":False,
	"lam":{0:0.0,1:0.0,2:0.0,3:0.0,4:0.0},
	"order":"N"}

Bd = {"atoms":[0,1,2,3,4],
	 "Z":{0:83,1:83,2:34,3:34,4:34,5:34},
	 "pos":[np.array([0.0000,0.0000,11.4114]),np.array([0.0000,0.0000,17.2245]),np.array([0.0000,0.0000,0.0000]),np.array([0.0000,0.0000,6.0565]),np.array([0.0000,0.0000,22.5795])],
	"spin":spin_dict}

