# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 14:33:53 2016

@author: rday
"""
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

#a=3.7734/np.sqrt(2) #FeSe
a = 3.7914/np.sqrt(2) #LiFeAs
#c = 5.5258
c = 6.3639
olist = ["xy","XY","xz","yz","ZR","xy","XY","xz","yz","ZR","xy","XY","xz","yz","ZR","xy","XY","xz","yz","ZR"]

k1 = [a,a,0]
k2 = [-a,a,0]
kx = [a,0,0]
ky = [0,a,0]
Hpp = [""]*13
Hpm = [""]*11
k_args = {"k1":np.array([a,a,0]),"k2":np.array([-a,a,0]),"kx":np.array([a,0,0]),
          "ky":np.array([0,a,0]),"2_kx":np.array([2*a,0,0]),
          "2_ky":np.array([0,2*a,0]),"k2y":np.array([-a,2*a,0]),
          "k1y":np.array([a,2*a,0]),"k2x":np.array([-2*a,a,0]),
          "k1x":np.array([2*a,a,0]),"kz":np.array([0,0,c])}
          
signdict = {"+":1.0,"-":-1.0}
def rot(T,v):
    mat = np.array([[np.cos(T),-np.sin(T),0.0],[np.sin(T),np.cos(T),0.0],[0.0,0.0,1.0]])
    v2 = np.dot(mat,v)
    for t in range(3):
        v2[t]=round(v2[t],5)
    return v2
    
klist = ["k1","k2","kx","ky","2_kx","2_ky","k2y","k1y","k2x","k1x","kz"]
#for i in range(len(klist)):
#    tmp = rot(np.pi/4.,k_args[klist[i]])
#    k_args[klist[i]] = tmp

#FeSe
shift = 0.0
off =0.0*2.17# 0.005*2.17#0.03*2.17

##FeSe E&K
#tdic = {"t11_11":0.086,"t10_16":-0.063,"t20_11":-0.028,"t21_16":0.017,"t11_13":-0.056*1.0j,
#        
#"t10_18":0.305*1.0j,"t11_15":-0.109,"t10_27":-0.412,"E1":0.014-shift,"t11_22":-0.066,"t10_29":-0.364*1.0j,
#
#"E2":-0.539-shift,"t11_23":0.089*1.0j,"t10_210":0.338,"E3":0.02-shift,"t11_33":0.232,"t10_38":0.08,
#
#"E5":-0.581-shift,"t20_33":0.009,"t21_38":0.016,"t02_33":-0.045,"t10_49":0.311,"t22_33":0.027,
#
#"t21_49":-0.019,"t11_34":0.099,"t10_410":0.180*1.0j,"t11_35":0.146*1.0j,
#
#"t101_16":0.0,"t001_11":0.0,"t121_16":-0.017,"t111_11":0.0,"t101_18":0.009*1.0j,
#
#"t201_11":0.017,"t101_19":0.020*1.0j,"t201_14":0.030*1.0j,"t211_19":0.031*1.0j,
#
#"t001_33":0.011,"t101_38":0.006,"t201_33":-0.008,"t121_38":-0.003,"t021_33":0.02,
#
#"t101_39":0.015,"t101_49":0.025,"t121_49":0.006}
###tdic["dxz"]=tdic["E3"]+0.1
#LiFeAs ARPES
#tdic = {"E1":0.02-off,"E2":-0.2605-off,"E3":-0.0075-off,"E5":-0.3045-off,"t11_11":0.03,"t10_16":-0.0185,
#        
#        "t20_11":-0.01,"t21_16":0.0035,"t11_13":-0.0635*1.0j,"t10_18":0.155*1.0j,
#        
#        "t11_15":-0.09,"t10_27":-0.2225,"t11_22":0.07,"t10_29":-0.1925*1.0j,
#        
#        "t11_23":-0.010*1.0j,"t10_210":0.1615,"t11_33":0.152,"t10_38":0.050,
#        
#        "t20_33":-0.004,"t21_38":0.04,"t02_33":-0.051,"t10_49":0.210,"t22_33":-0.005,
#        
#        "t21_49":-0.053,"t11_34":0.09,"t10_410":0.0995*1.0j,"t11_35":0.1005*1.0j,
#        
#        "t101_16":-0.004,"t001_11":0.0105,"t111_11":0.0,"t201_11":0.0,"t201_14":0.0,
#        
#        "t001_33":-0.003,"t201_33":0.0,"t021_33":0.0105,"t121_16":0.0,"t101_18":0.0,
#        
#        "t101_19":0.0,"t211_19":0.0,"t101_38":0.0115,"t121_38":0.0,"t101_39":0.0,
#        
#        "t101_49":0.0,"t121_49":0.0}
#LiFeAs Koepernik and Eschrig
tdic = {"t11_11":0.079,"t10_16":-0.016,"t20_11":0.02,"t21_16":0.013,"t11_13":-0.090j,
        
        "t10_18":0.281j,"t11_15":-0.06,"t10_27":-0.404,"E1":-0.188-off,"t11_22":-0.032,
        
        "t10_29":-0.353j,"E2":-0.521-off,"t11_23":0.087j,"t10_210":0.313,"E3":0.2-off,
        
        "t11_33":0.275,"t10_38":0.125,"E5":-0.609-off,"t20_33":-0.002,"t21_38":0.056,
        
        "t02_33":-0.107,"t10_49":0.359,"t22_33":0.012,"t21_49":-0.048,"t11_34":0.102,
        
        "t10_410":0.190j,"t11_35":0.136j,"t101_16":-0.018,"t001_11":0.07,"t121_16":-0.025,
        
        "t111_11":0.02,"t101_18":0.008j,"t201_11":0.005,"t101_19":0.02j,"t201_14":0.025j,
        
        "t211_19":0.022j,"t001_33":-0.004,"t101_38":0.0,"t201_33":-0.003,"t121_38":-0.014,
        
        "t021_33":0.031,"t101_39":0.017,"t101_49":0.016,"t121_49":0.043}
##LiFeAs Giorgio 2 (fit to ARPES (after renormalizing by 2.17))



#######

#tdic = {"t11_11":0.059,"t10_16":-0.016,"t20_11":0.02,"t21_16":0.008,"t11_13":-0.09j,
#        
#        "t10_18":0.23j,"t11_15":-0.025,"t10_27":-0.404,"E1":-0.173-off,"t11_22":-0.032,
#        
#        "t10_29":-0.373j,"E2":-0.521-off,"t11_23":0.052j,"t10_210":0.403,"E3":0.2-off,"t11_33":0.255,
#        
#        "t10_38":0.095,"E5":-0.904-off,"t20_33":-0.002,"t21_38":0.056,"t02_33":-0.107,
#        
#        "t10_49":0.359,"t22_33":0.012,"t21_49":-0.048,"t11_34":0.022,"t10_410":0.19j,
#        
#        "t11_35":0.116j,"t101_16":-0.018*kz_renorm,"t001_11":0.07*kz_renorm,"t121_16":-0.025*kz_renorm,"t111_11":0.02*kz_renorm,
#        
#        "t101_18":0.008j*kz2,"t201_11":0.005*kz_renorm,"t101_19":0.02j*kz2,"t201_14":0.025j*kz2,"t211_19":0.022j*kz2,
#        
#        "t001_33":-0.004*kz_01,"t101_38":0.0,"t201_33":-0.003*kz_21,"t121_38":-0.014*kz_31,
#        
#        "t021_33":0.031*kz_21,"t101_39":0.017*kz_11,"t101_49":0.016*kz_11,"t121_49":0.043*kz_31} 
##LiFeAs Giorgio 1   (adapted from E&K)     
#tdic = {"t11_11":0.079,"t10_16":-0.016,"t20_11":0.02,"t21_16":0.013,"t11_13":-0.09j,
#        
#        "t10_18":0.281j,"t11_15":-0.06,"t10_27":-0.404,"E1":-0.188,"t11_22":-0.032,
#        
#        "t10_29":-0.353j,"E2":-0.521,"t11_23":0.087j,"t10_210":0.313,"E3":0.2,
#        
#        "t11_33":0.275,"t10_38":0.125,"E5":-0.609,"t20_33":-0.002,"t21_38":0.056,
#        
#        "t02_33":-0.107,"t10_49":0.359,"t22_33":0.012,"t21_49":-0.048,"t11_34":0.102,
#        
#        "t10_410":0.19j,"t11_35":0.136j,"t101_16":-0.018,"t001_11":0.07,"t121_16":-0.025,
#        
#        "t111_11":0.02,"t101_18":0.008j,"t201_11":0.005,"t101_19":0.02j,"t201_14":0.025j,
#        
#        "t211_19":0.022j,"t001_33":-0.004,"t101_38":0.0,"t201_33":-0.003,"t121_38":-0.014,
#        
#        "t021_33":0.031,"t101_39":0.017,"t101_49":0.016,"t121_49":0.043}


        
Hpp[0] = "1,1,E1,2*t11_11*cos(k1),2*t11_11*cos(k2),2*t20_11*cos(2_kx),2*t20_11*cos(2_ky)"
Hpp[1]= "1,3,2.0j*t11_13*sin(k1),-2.0j*t11_13*sin(k2)"
Hpp[2] = "1,4,2.0j*t11_13*sin(k1),2.0j*t11_13*sin(k2)"
Hpp[3] = "1,5,2*t11_15*cos(k1),-2*t11_15*cos(k2)"
Hpp[4] = "2,2,E2,2*t11_22*cos(k1),2*t11_22*cos(k2)"
Hpp[5] = "2,3,2.0j*t11_23*sin(k1),2.0j*t11_23*sin(k2)"
Hpp[6] = "2,4,-2.0j*t11_23*sin(k1),2.0j*t11_23*sin(k2)"
Hpp[7] = "3,3,E3,2*t11_33*cos(k1),2*t11_33*cos(k2),2*t20_33*cos(2_kx),2*t02_33*cos(2_ky),4*t22_33*cos(2_kx)*cos(2_ky)"
Hpp[8] = "3,4,2*t11_34*cos(k1),-2*t11_34*cos(k2)"
Hpp[9] = "3,5,2.0j*t11_35*sin(k1),2.0j*t11_35*sin(k2)"
Hpp[10] = "4,4,E3,2*t11_33*cos(k1),2*t11_33*cos(k2),2*t02_33*cos(2_kx),2*t20_33*cos(2_ky),4*t22_33*cos(2_kx)*cos(2_ky)"
Hpp[11] = "4,5,2.0j*t11_35*sin(k1),-2.0j*t11_35*sin(k2)"
Hpp[12] = "5,5,E5"

Hpm[0] = "1,6,2*t10_16*cos(kx),2*t10_16*cos(ky),2*t21_16*cos(k1)*cos(kx),2*t21_16*cos(k1)*cos(ky),2*t21_16*cos(k2)*cos(kx),2*t21_16*cos(k2)*cos(ky),-2*t21_16*sin(k1)*sin(kx),-2*t21_16*sin(k1)*sin(ky),2*t21_16*sin(k2)*sin(kx),-2*t21_16*sin(k2)*sin(ky)"
Hpm[1] = "1,8,2.0j*t10_18*sin(kx)"
Hpm[2] = "1,9,2.0j*t10_18*sin(ky)"
Hpm[3] = "2,7,2*t10_27*cos(kx),2*t10_27*cos(ky)"
Hpm[4] = "2,8,-2.0j*t10_29*sin(ky)"
Hpm[5] = "2,9,2.0j*t10_29*sin(kx)"
Hpm[6] = "2,10,2*t10_210*cos(kx),-2*t10_210*cos(ky)"
Hpm[7] = "3,8,2*t10_38*cos(kx),2*t10_49*cos(ky),2*t21_38*cos(k1)*cos(kx),2*t21_38*cos(k2)*cos(kx),-2*t21_38*sin(k1)*sin(kx),2*t21_38*sin(k2)*sin(kx),2*t21_49*cos(k1)*cos(ky),2*t21_49*cos(k2)*cos(ky),-2*t21_49*sin(k1)*sin(ky),-2*t21_49*sin(k2)*sin(ky)"
Hpm[8] = "3,10,2.0j*t10_410*sin(ky)"
Hpm[9] = "4,9,2*t10_49*cos(kx),2*t10_38*cos(ky),2*t21_49*cos(k1)*cos(kx),2*t21_49*cos(k2)*cos(kx),-2*t21_49*sin(k1)*sin(kx),2*t21_49*sin(k2)*sin(kx),2*t21_38*cos(k1)*cos(ky),2*t21_38*cos(k2)*cos(ky),-2*t21_38*sin(k1)*sin(ky),-2*t21_38*sin(k2)*sin(ky)"
Hpm[10] = "4,10,2.0j*t10_410*sin(kx)"

#3D part
Hpp3 = [""]*5
Hpm3 = [""]*6
Hpp3[0] = "1,1,2*t001_11*cos(kz),4*t111_11*cos(k1)*cos(kz),4*t111_11*cos(k2)*cos(kz),4*t201_11*cos(2_kx)*cos(kz),4*t201_11*cos(2_ky)*cos(kz)"
Hpp3[1] = "1,3,-4*t201_14*sin(2_ky)*sin(kz)"
Hpp3[2] = "1,4,-4*t201_14*sin(2_kx)*sin(kz)"
Hpp3[3] = "3,3,2*t001_33*cos(kz),4*t201_33*cos(2_kx)*cos(kz),4*t021_33*cos(2_ky)*cos(kz)"
Hpp3[4] = "4,4,2*t001_33*cos(kz),4*t021_33*cos(2_kx)*cos(kz),4*t201_33*cos(2_ky)*cos(kz)"
Hpm3[0] = "1,6,4.0*t101_16*cos(kx)*cos(kz),4*t101_16*cos(ky)*cos(kz),2*t121_16*cos(k1y)*exp(+kz),2*t121_16*cos(k1x)*exp(+kz),2*t121_16*cos(k2y)*exp(-kz),2*t121_16*cos(k2x)*exp(-kz)"
Hpm3[1] = "1,8,4.0j*t101_18*sin(kx)*cos(kz),-4*t101_19*sin(ky)*sin(kz),2.0j*t211_19*sin(k1y)*exp(+kz),-2.0j*t211_19*sin(k2y)*exp(-kz)"
Hpm3[2] = "1,9,4.0j*t101_18*sin(ky)*cos(kz),-4*t101_19*sin(kx)*sin(kz),2.0j*t211_19*sin(k1x)*exp(+kz),2.0j*t211_19*sin(k2x)*exp(-kz)"
Hpm3[3] = "3,8,4*t101_38*cos(kx)*cos(kz),4*t101_49*cos(ky)*cos(kz),2*t121_38*cos(k1x)*exp(+kz),2*t121_38*cos(k2x)*exp(-kz),2*t121_49*cos(k1y)*exp(+kz),2*t121_49*cos(k2y)*exp(-kz)"
Hpm3[4] = "3,9,4.0j*t101_39*cos(kx)*sin(kz),4.0j*t101_39*cos(ky)*sin(kz)" 
Hpm3[5] = "4,9,4*t101_49*cos(kx)*cos(kz),4*t101_38*cos(ky)*cos(kz),2*t121_49*cos(k1x)*exp(+kz),2*t121_49*cos(k2x)*exp(-kz),2*t121_38*cos(k1y)*exp(+kz),2*t121_38*cos(k2y)*exp(-kz)"


#Hpm3[1] = "1,8,-4.0*t101_18*sin(kx)*cos(kz),-4.0*t101_19*sin(ky)*sin(kz),2.0j*t211_19*sin(k1y)*exp(+kz),-2.0j*t211_19*sin(k2y)*exp(-kz)"
#Hpm3[2] = "1,9,-4.0*t101_18*sin(ky)*cos(kz),-4.0*t101_19*sin(kx)*sin(kz),2.0j*t211_19*sin(k1x)*exp(+kz),2.0j*t211_19*sin(k2x)*exp(-kz)"

basis = ["3dxy","3dXY","1.0j*3dxz","1.0j*3dyz","3dZR","3dxy","3dXY","-1.0j*3dxz","-1.0j*3dyz","3dZR"]
S = 1.0# for LiFeAs
#def transform(Helements):
#    tmp = []
#    tmpS = np.identity(10,dtype=complex)
#    inds = np.array([1.0,1.0,1.0j,1.0j,1.0,1.0,1.0,-1.0j,-1.0j,1.0],dtype=complex)
#  #  inds = np.array([1.0,1.0,-1.0j,-1.0j,1.0,1.0,1.0,1.0j,1.0j,1.0],dtype=complex)
#  #  inds = np.array([1.0,1.0,-1.0j,-1.0j,1.0,1.0,1.0,1.0j,1.0j,1.0],dtype=complex
#
#    S = tmpS*inds
#    Sp = np.linalg.inv(S)
#    for h in list(enumerate(Helements)):
#        new_val=h[1]
#        Htmp=np.zeros((10,10),dtype=complex)
#        Htmp[int(h[1][0]),int(h[1][1])]=complex(h[1][-1])#*inds[int(h[1][1])]
#        res = np.dot(Sp,np.dot(Htmp,S))
#       # res = np.dot(Sp,Htmp)
#        for i in range(10):
#            for j in range(10):
#                if res[i,j]!=0.0:
#                    new_val[-1] = str(res[i,j])
#                    new_val[0]=str(i)
#                    new_val[1]=str(j)
#        tmp.append(new_val)
#    return tmp

def transform(Helements):
    tmp = []
    S = np.zeros((10,10),dtype=complex)
#    inds = np.array([1.0,1.0,1.0j,-1.0,1.0,1.0,1.0,-1.0j,1.0,1.0],dtype=complex)
    inds = np.array([1.0,1.0,1.0j,1.0j,1.0,1.0,1.0,-1.0j,-1.0j,1.0],dtype=complex)
#    inds = np.ones(10,dtype=complex)
#    inds = np.array([1.0,1.0,1.0j,1.0j*np.exp(np.pi/4),1.0,1.0,1.0,-1.0j,-1.0j*np.exp(np.pi/4),1.0],dtype=complex)

#    inds = np.ones(10,dtype=complex)
    for i in range(10):    
        S[i,i]=inds[i]
    Sp = np.linalg.inv(S)
    for h in list(enumerate(Helements)):
        new_val=h[1]
        Htmp=np.zeros((10,10),dtype=complex)
        Htmp[int(h[1][0]),int(h[1][1])]=complex(h[1][-1])*inds[int(h[1][1])]
        res = np.dot(Sp,Htmp)
        for i in range(10):
            for j in range(10):
                if res[i,j]!=0.0:
                    new_val[-1] = str(res[i,j])
                    new_val[0]=str(i)
                    new_val[1]=str(j)
        tmp.append(new_val)
    return tmp         
        

def Hparse(Hstring,tdic):
    all_elements = []
    tmp = Hstring.split(",")
    ind1 = str(int(tmp[0])-1)
    ind2 = str(int(tmp[1])-1)
    for i in range(2,len(tmp)):
        raw = tmp[i]
        if raw[0]=="E" and not raw[1:].isalpha():
            all_elements.append([ind1,ind2,np.array([0.0,0.0,0.0]),tdic[raw]/S])
        else:
            raw_split = raw.split("*")
            tmp2 = complex(raw_split[0])*tdic[raw_split[1]]/S
            v = []
            if raw_split[2][:3]=="cos":
                v.append([0.5*tmp2,k_args[raw_split[2][4:-1]]])
                v.append([0.5*tmp2,-k_args[raw_split[2][4:-1]]])
            elif raw_split[2][:3]=="sin":
                v.append([0.5j*tmp2,-k_args[raw_split[2][4:-1]]])
                v.append([-0.5j*tmp2,k_args[raw_split[2][4:-1]]])  
            if raw_split[2][:3]!="cos" and raw_split[2][:3]!="sin":
                print('help term 1',tmp)
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
                elif raw_split[3][:3]!="cos" and raw_split[3][:3]!="sin" and raw_split[3][:3]!="exp":
                    print('help term 2!',tmp)
            for j in range(len(v)):
                all_elements.append([ind1,ind2,np.array(v[j][1]),v[j][0]])
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
    
def fillH(A,B,tdic):
    Hlist = []
    for i in range(len(A)):#iterate through H++--these will be near identical for atom 1 and atom2, exception being H++_2 = H++_1*
        tmp_list = Hparse(A[i],tdic)
        for j in range(len(tmp_list)):
            sub_tmp = tmp_list[j]  
            if abs(complex(sub_tmp[-1]))!=0.0:
                Hlist.append([sub_tmp[0],sub_tmp[1],str(sub_tmp[2][0]),str(sub_tmp[2][1]),str(sub_tmp[2][2]),str(sub_tmp[3])]) 
                Hlist.append([str(int(sub_tmp[0])+5),str(int(sub_tmp[1])+5),str(-sub_tmp[2][0]),str(-sub_tmp[2][1]),str(-sub_tmp[2][2]),str(np.conj(sub_tmp[3]))]) 
    for i in range(len(B)): #compute Hpm
        tmp_list = Hparse(B[i],tdic)
        for j in range(len(tmp_list)):
            sub_tmp = tmp_list[j]
            if abs(complex(sub_tmp[-1]))!=0.0:
                Hlist.append([sub_tmp[0],sub_tmp[1],str(sub_tmp[2][0]),str(sub_tmp[2][1]),str(sub_tmp[2][2]),str(sub_tmp[3])])             
            if int(sub_tmp[1])-5>int(sub_tmp[0]):
                sub_tmp2 = [str(int(sub_tmp[1])-5),str(int(sub_tmp[0])+5),str(sub_tmp[2][0]),str(sub_tmp[2][1]),str(sub_tmp[2][2]),str(sub_tmp[3])]    
                if abs(complex(sub_tmp[-1]))!=0.0:                
                    Hlist.append(sub_tmp2)
    return Hlist
    
def spin_copy(Hlist,dim):
    tmp = [[str(int(h[0])+dim),str(int(h[1])+dim),h[2],h[3],h[4],h[5]] for h in Hlist]
    return tmp
def spin_orbit(olist,L):
    SO = {}
    H_SOC=[]
    dim = len(olist)
    for i in range(dim):
        for j in range(dim):
            if i%(dim/2)/(dim/4)!=j%(dim/2)/(dim/4): #same site
                continue
            elif i%(dim/2)/(dim/4)==j%(dim/2)/(dim/4):
                if i<=j:
                    orb_string = olist[i]+olist[j]
                    s1 = 2*(i/10)-1 #down spin first
                    s2 = 2*(j/10)-1 #up spin last
                    ds = (s1-s2)/2.0
                    s = np.zeros(3)
                    s[int(ds)+1] = 1.0/2.0 #expectation of spin operators: 0.5*s-,s_z,0.5*s+
                    if s1==-1 and s2==-1:
                        s[1]=-0.5
                    V = L*np.array([[2.0*s[1],2.0*s[0],0,0,0],[2.0*s[2],s[1],np.sqrt(6)*s[0],0,0],[0,np.sqrt(6)*s[2],0,np.sqrt(6)*s[0],0],[0,0,np.sqrt(6)*s[2],-s[1],2*s[0]],[0,0,0,2*s[2],-2*s[1]]])                   
                    Mp = np.sqrt(0.5)*np.array([[0,0,-1.0j,0,1.0],[1.0j,-1.0,0,0,0],[0,0,0,np.sqrt(2),0],[1.0j,1.0,0,0,0],[0,0,1.0j,0,1.0]])
                    Mpp = np.linalg.inv(Mp)
                    Vp = np.dot(Mpp,np.dot(V,Mp))

                    SO["yzyz"],SO["yzxz"],SO["yzxy"],SO["yzZR"],SO["yzXY"]=Vp[0]
                    SO["xzyz"],SO["xzxz"],SO["xzxy"],SO["xzZR"],SO["xzXY"]=Vp[1]
                    SO["xyyz"],SO["xyxz"],SO["xyxy"],SO["xyZR"],SO["xyXY"]=Vp[2]
                    SO["ZRyz"],SO["ZRxz"],SO["ZRxy"],SO["ZRZR"],SO["ZRXY"]=Vp[3]
                    SO["XYyz"],SO["XYxz"],SO["XYxy"],SO["XYZR"],SO["XYXY"]=Vp[4]                 
                    SO_append = [str(i),str(j),"0.0","0.0","0.0",str(SO[orb_string])]
                    if abs(complex(SO_append[-1]))>10**-6:
                        H_SOC.append(SO_append)
    return H_SOC
    
def writefile(H,filename,mode):
    with open(filename,mode) as tofile:
        for i in range(len(H)):
            wl = ""
            for j in H[i]:
                wl += j+","
            wl +="\n"
            tofile.write(wl)
        tofile.close
        
        
def make_Ham(args):
    
    
    xy = args[0]
    xzyz = args[1]
    xyxz = args[2]
    xzyz2 = args[3] 
    rshift = 0.035
    
#    tdic = {"t11_11":0.059,"t10_16":-0.016,"t20_11":0.02,"t21_16":0.008,"t11_13":-0.09j,
#            
#            "t10_18":0.23j,"t11_15":-0.025,"t10_27":-0.404,"E1":-0.173+rshift-off,"t11_22":-0.032,
#            
#            "t10_29":-0.403j,"E2":-0.521-off,"t11_23":0.052j,"t10_210":0.403,"E3":0.2-off,"t11_33":0.255,
#            
#            "t10_38":0.095,"E5":-0.904-off,"t20_33":-0.002,"t21_38":0.056,"t02_33":-0.107,
#        
#            "t10_49":0.359,"t22_33":0.002,"t21_49":-0.048,"t11_34":0.022,"t10_410":0.19j,
#            
#            "t11_35":0.116j,"t101_16":-0.018*xy,"t001_11":0.07*xy,"t121_16":-0.025*xy,"t111_11":0.02*xy,
#            
#            "t101_18":0.008j*xyxz,"t201_11":0.005*xy,"t101_19":0.02j*xy,"t201_14":0.025j*xyxz,"t211_19":0.022j*xy,
#            
#            "t001_33":-0.004*xzyz,"t101_38":0.0*xzyz,"t201_33":-0.003*xzyz,"t121_38":-0.014*xzyz,
#            
#            "t021_33":0.031*xzyz2,"t101_39":0.017*xzyz2,"t101_49":0.016*xzyz2,"t121_49":0.043*xzyz2} 


    
    Hlist = fillH(Hpp,Hpm,tdic)  
    Hlist3D = fillH(Hpp3,Hpm3,tdic)
    Htrans_d=transform(Hlist)
    H3Dtrans_d=transform(Hlist3D) #convert to real orbitals since this model uses idxz and idyz
#    Htrans_u = spin_copy(Hlist,10)
#    H3Dtrans_u = spin_copy(Hlist3D,10) 
#    for i in range(11):
#    H_SOC=spin_orbit(olist,0.011*f)
    filename = "C:\\Users\\rday\\Documents\\TB_ARPES\\2018\\TB_ARPES_2018\\LiFeAs_EK.txt"
    writefile(Htrans_d,filename,"w")
    writefile(H3Dtrans_d,filename,"a")
    #writefile(Htrans_u,filename,"a")
    #writefile(H3Dtrans_u,filename,"a")
    #writefile(H_SOC,filename,"a")
    print('complete')
    
    
if __name__ == "__main__":
#    args = [1.2,1.5,1.3,1.]

    args = [1.2,2.1,1.3,0.9]
#    args = [0,0,0,0]
    make_Ham(args)










#    tdic = {"t11_11":0.059,"t10_16":-0.016,"t20_11":0.02,"t21_16":0.008,"t11_13":-0.09j,
#            
#            "t10_18":0.23j,"t11_15":-0.025,"t10_27":-0.404,"E1":-0.173+rshift-off,"t11_22":-0.032,
#            
#            "t10_29":-0.373j,"E2":-0.521-off,"t11_23":0.052j,"t10_210":0.403,"E3":0.2-off,"t11_33":0.255,
#            
#            "t10_38":0.095,"E5":-0.904-off,"t20_33":-0.002,"t21_38":0.056,"t02_33":-0.107,
#        
#            "t10_49":0.359,"t22_33":0.012,"t21_49":-0.048,"t11_34":0.022,"t10_410":0.19j,
#            
#            "t11_35":0.116j,"t101_16":-0.018*xy,"t001_11":0.07*xy,"t121_16":-0.025*xy,"t111_11":0.02*xy,
#            
#            "t101_18":0.008j*xyxz,"t201_11":0.005*xy,"t101_19":0.02j*xy,"t201_14":0.025j*xyxz,"t211_19":0.022j*xy,
#            
#            "t001_33":-0.004*xzyz,"t101_38":0.0*xzyz,"t201_33":-0.003*xzyz,"t121_38":-0.014*xzyz,
#            
#            "t021_33":0.031*xzyz2,"t101_39":0.017*xzyz2,"t101_49":0.016*xzyz2,"t121_49":0.043*xzyz2} 