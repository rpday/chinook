# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 14:33:53 2016

@author: rday
"""
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

a=5.839/np.sqrt(2) #FeSe
c = 9.321
#c = 6.3639
k1 = [a,a,0]
k2 = [-a,a,0]
kx = [a,0,0]
ky = [0,a,0]
Hpp = [""]*15

k_args = {"kx":np.array([a,0,0]),"ky":np.array([0,a,0]),
         "2_kx":np.array([2*a,0,0]),"2_ky":np.array([0,2*a,0])}

          
signdict = {"+":1.0,"-":-1.0}
def rot(T,v):
    mat = np.array([[np.cos(T),-np.sin(T),0.0],[np.sin(T),np.cos(T),0.0],[0.0,0.0,1.0]])
    v2 = np.dot(mat,v)
    return v2


#FeSe
tdic = {"E1":0.13,"E3":-0.22,"E4":0.3,"E5":-0.21,
        "t11_x":-0.14,"t11_y":-0.4,"t11_xy":0.28,"t11_xx":0.02,"t11_xxy":-0.035,"t11_xyy":0.005,"t11_xxyy":0.035,
        "t33_x":0.35,"t33_y":0.0,"t33_xy":-0.105,"t33_xx":-0.02,"t33_xxy":0.0,"t33_xyy":0.0,"t33_xxyy":0.,
        "t44_x":0.23,"t44_y":0.0,"t44_xy":0.15,"t44_xx":-0.03,"t44_xxy":-0.03,"t44_xyy":0.0,"t44_xxyy":-0.03,
        "t55_x":-0.1,"t55_y":0.0,"t55_xy":0.0,"t55_xx":-0.04,"t55_xxy":0.02,"t55_xyy":0.0,"t55_xxyy":-0.01,
        "t12_x":0.,"t12_xy":0.05,"t12_xxy":-0.015,"t12_xxyy":0.035,
        "t13_x":-0.354,"t13_xy":0.099,"t13_xxy":0.021,"t13_xxyy":0.0,
        "t14_x":0.339,"t14_xy":0.014,"t14_xxy":0.028,"t14_xxyy":0.0,
        "t15_x":-0.198,"t15_xy":-0.085,"t15_xxy":0.0,"t15_xxyy":-0.014,
        "t34_x":0.0,"t34_xy":0,"t34_xxy":-0.01,"t34_xxyy":0.0,
        "t35_x":-0.3,"t35_xy":0.0,"t35_xxy":-0.02,"t35_xxyy":0,
        "t45_x":0.0,"t45_xy":-0.15,"t45_xxy":0.0,"t45_xxyy":0.01}


Hpp[0] = "1,1,E1,2.0*t11_x*cos(kx),2*t11_y*cos(ky),4*t11_xy*cos(kx)*cos(ky),2*t11_xx*cos(2_kx),-2.0*t11_xx*cos(2_ky),4*t11_xxy*cos(2_kx)*cos(ky),4*t11_xyy*cos(2_ky)*cos(kx),4*t11_xxyy*cos(2_kx)*cos(2_ky)"

Hpp[1] = "2,2,E1,2.0*t11_y*cos(kx),2*t11_x*cos(ky),4*t11_xy*cos(kx)*cos(ky),-2.0*t11_xx*cos(2_kx),2.0*t11_xx*cos(2_ky),4*t11_xyy*cos(2_kx)*cos(ky),4*t11_xxy*cos(2_ky)*cos(kx),4*t11_xxyy*cos(2_kx)*cos(2_ky)"

Hpp[2] = "3,3,E3,2.0*t33_x*cos(kx),2.0*t33_x*cos(ky),4*t33_xy*cos(kx)*cos(ky),2*t33_xx*cos(2_kx),2*t33_xx*cos(2_ky)"

Hpp[3]= "4,4,E4,2*t44_x*cos(kx),2*t44_x*cos(ky),4*t44_xy*cos(kx)*cos(ky),2*t44_xx*cos(2_kx),2*t44_xx*cos(2_ky),4*t44_xxy*cos(2_kx)*cos(ky),4*t44_xxy*cos(2_ky)*cos(kx),4*t44_xxyy*cos(2_kx)*cos(2_ky)"

Hpp[4] = "5,5,E5,2*t55_x*cos(kx),2*t55_x*cos(ky),2*t55_xx*cos(2_kx),2*t55_xx*cos(2_ky),4*t55_xxy*cos(2_kx)*cos(ky),4*t55_xxy*cos(2_ky)*cos(kx),4*t55_xxyy*cos(2_kx)*cos(2_ky)"

Hpp[5] = "1,2,-4*t12_xy*sin(kx)*sin(ky),-4*t12_xxy*sin(2_kx)*sin(ky),-4*t12_xxy*sin(2_ky)*sin(kx),-4*t12_xxyy*sin(2_kx)*sin(2_ky)"

Hpp[6] = "1,3,2.0j*t13_x*sin(ky),4.0j*t13_xy*sin(ky)*cos(kx),-4.0j*t13_xxy*sin(2_ky)*cos(kx),4.0j*t13_xxy*cos(2_kx)*sin(ky)"

Hpp[7] = "2,3,-2.0j*t13_x*sin(kx),-4.0j*t13_xy*sin(kx)*cos(ky),4.0j*t13_xxy*sin(2_kx)*cos(ky),-4.0j*t13_xxy*cos(2_ky)*sin(kx)"

Hpp[8] = "1,4,2.0j*t14_x*sin(kx),4.0j*t14_xy*cos(ky)*sin(kx),4.0j*t14_xxy*sin(2_kx)*cos(ky)"

Hpp[9] = "2,4,2.0j*t14_x*sin(ky),4.0j*t14_xy*cos(kx)*sin(ky),4.0j*t14_xxy*sin(2_ky)*cos(kx)"

Hpp[10] = "1,5,2.0j*t15_x*sin(ky),-4.0j*t15_xy*sin(ky)*cos(kx),-4.0j*t15_xxyy*sin(2_ky)*cos(2_kx)"

Hpp[11] = "2,5,2.0j*t15_x*sin(kx),-4.0j*t15_xy*sin(kx)*cos(ky),-4.0j*t15_xxyy*sin(2_kx)*cos(2_ky)"

Hpp[12] = "3,4,4*t34_xxy*sin(2_ky)*sin(kx),-4*t34_xxy*sin(2_kx)*sin(ky)"

Hpp[13] = "3,5,2*t35_x*cos(kx),-2*t35_x*cos(ky),4*t35_xxy*cos(2_kx)*cos(ky),-4*t35_xxy*cos(2_ky)*cos(kx)"

Hpp[14] = "4,5,4*t45_xy*sin(kx)*sin(ky),4*t45_xxyy*sin(2_kx)*sin(2_ky)"



basis = ["3dxz","3dyz","3dXY","3dxy","3dZR"]

        

def Hparse(Hstring):
    all_elements = []
    tmp = Hstring.split(",")
    ind1 = str(int(tmp[0])-1)
    ind2 = str(int(tmp[1])-1)
    for i in range(2,len(tmp)):
        raw = tmp[i]
        if raw[0]=="E" and not raw[1:].isalpha():
            all_elements.append([ind1,ind2,np.array([0.0,0.0,0.0]),tdic[raw]])
        else:
            raw_split = raw.split("*")
            tmp2 = complex(raw_split[0])*tdic[raw_split[1]]
            v = []
            if raw_split[2][:3]=="cos":
                v.append([0.5*tmp2,k_args[raw_split[2][4:-1]]])
                v.append([0.5*tmp2,-k_args[raw_split[2][4:-1]]])
            elif raw_split[2][:3]=="sin":
                v.append([0.5j*tmp2,-k_args[raw_split[2][4:-1]]])
                v.append([-0.5j*tmp2,k_args[raw_split[2][4:-1]]])  
            if len(raw_split)>3:
                tmp_v = v
                v = []
                if raw_split[3][:3]=="cos":
                    v.append([0.5*tmp_v[0][0],tmp_v[0][1]+k_args[raw_split[3][4:-1]]])
                    v.append([0.5*tmp_v[1][0],tmp_v[1][1]+k_args[raw_split[3][4:-1]]])
                    v.append([0.5*tmp_v[0][0],tmp_v[0][1]-k_args[raw_split[3][4:-1]]])
                    v.append([0.5*tmp_v[1][0],tmp_v[1][1]-k_args[raw_split[3][4:-1]]])
                elif raw_split[3][:3]=="sin":
                    v.append([-0.5j*tmp_v[0][0],tmp_v[0][1]+k_args[raw_split[3][4:-1]]])
                    v.append([-0.5j*tmp_v[1][0],tmp_v[1][1]+k_args[raw_split[3][4:-1]]])
                    v.append([0.5j*tmp_v[0][0],tmp_v[0][1]-k_args[raw_split[3][4:-1]]])
                    v.append([0.5j*tmp_v[1][0],tmp_v[1][1]-k_args[raw_split[3][4:-1]]])
                elif raw_split[3][:3]=="exp":
                    sign = signdict[raw_split[3][4]]
                    v_new = sign*k_args[raw_split[3][5:-1]]
                    v.append([tmp_v[0][0],tmp_v[0][1]+v_new])
                    v.append([tmp_v[1][0],tmp_v[1][1]+v_new])
            for i in range(len(v)):
                all_elements.append([ind1,ind2,np.array(v[i][1]),v[i][0]])
    simplified = []
    kill_list = [False]*len(all_elements)
    for a in list(enumerate(all_elements)):
        count = 1
        for b in list(enumerate(all_elements)):
            if a[0]!=b[0]:
                if np.linalg.norm(a[1][2]-b[1][2])<0.0001 and a[1][3]==b[1][3]:
                    count+=1
                    kill_list[b[0]]=True
        if not kill_list[a[0]]:
            simplified.append([a[1][0],a[1][1],a[1][2],count*a[1][3]])
    return simplified
    
def fillH(A):
    Hlist = []
    for i in range(len(A)):
        tmp_list = Hparse(A[i])
        for j in range(len(tmp_list)):
            sub_tmp = tmp_list[j]           
            Hlist.append([sub_tmp[0],sub_tmp[1],str(sub_tmp[2][0]),str(sub_tmp[2][1]),str(sub_tmp[2][2]),str(sub_tmp[3])]) 

    return Hlist

def writefile(H,filename,mode):
    with open(filename,mode) as tofile:
        for i in range(len(H)):
            wl = ""
            for j in H[i]:
                wl += j+","
            wl +="\n"
            tofile.write(wl)
        tofile.close
      
if __name__=="__main__":
    Hlist = fillH(Hpp)  
    
    filename = "Graser.txt"
    writefile(Hlist,filename,"w")
    print('complete')