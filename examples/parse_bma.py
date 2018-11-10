# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 14:33:53 2016

@author: rday
"""
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
#import LiFeAs

a=3.7734/np.sqrt(2) #FeSe
#a = 3.7914/np.sqrt(2) #LiFeAs
c = 5.5258
#c = 6.3639
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

#
##FeSe Brian Andersen and PJ Hirschfeld
z = 6
#dt10_49,dt10_38,dt20_11,dt10_16,dt101_38,dt101_49=0.00016,0.00016,0.00066,0.00066,-0.00008,-0.00008
dt10_49,dt10_38,dt20_11,dt10_16,dt101_38,dt101_49 = 0,0,0,0,0,0

dz2 = -0.04
dxy = 0.018
dXY = 0.15
tdic = {'t11_11':0.086/z,'t10_16':-0.063/z-0.0211+dt10_16,
        't20_11':-0.028/z+0.0028+dt20_11,'t21_16':0.017/z,
        't11_13':-0.056j/z,'t10_18':0.305j/z+0.0762j,
        't11_15':-0.109/z,'t10_27':-0.412/z+0.02288,
        't11_22':-0.066/z+0.00366,'t10_29':-0.364j/z-0.03412j,
        't11_23':0.089j/z,'t10_210':0.338/z-0.01877,
        't11_33':0.232/z,'t10_38':0.08/z+0.00343+dt10_38,
        't20_33':0.009/z,'t21_38':0.016/z,
        't02_33':-0.045/z,'t10_49':0.311/z+0.0122+dt10_49,
        't22_33':0.027/z,'t21_49':-0.019/z,
        't11_34':0.099/z,'t10_410':0.18j/z,
        't11_35':0.146j/z,
        'E1':0.014/z+dxy,'E2':-0.539/z+0.02994+dXY,'E3':0.020/z+0.0,'E4':0.02/z+0.0,'E5':-0.581/z+0.03227+dz2,
        't101_16':0.00275,
        't001_11':0,'t121_16':-0.017/z,
        't111_11':0,'t101_18':0.009j/z,
        't201_11':0.017/z-0.00275,'t101_19':0.02j/z,
        't201_14':0.003j/z+0.0045j,'t211_19':-0.0031j/z,
        't001_33':0.011/z,'t101_38':0.006/z+0.00253+dt101_38,
        't201_33':-0.008/z,'t121_38':-0.003/z,
        't021_33':0.020/z,'t101_39':0.015/z,
        't101_49':0.025/z-0.00203+dt101_49,'t121_49':0.006/z
        }
###MY MOD of ANDERSEN MODEL
#dt10_49,dt10_38,dt20_11,dt10_16,dt101_38,dt101_49 = 0,0,0,0,0,0
#dz2,dXY = -0.12,-0.02
#dxy = 0.015
#tdic = {'t11_11':0.086/z,'t10_16':-0.063/z-0.0211+dt10_16,
#        't20_11':-0.028/z+0.0028+dt20_11,'t21_16':0.017/z,
#        't11_13':-0.056j/z,'t10_18':0.305j/z+0.0762j,
#        't11_15':-0.109/z,'t10_27':1.3*(-0.412/z+0.02288),
#        't11_22':-0.066/z+0.00366,'t10_29':-0.364j/z-0.03412j,
#        't11_23':0.089j/z,'t10_210':0.338/z-0.01877,
#        't11_33':0.232/z,'t10_38':0.08/z+0.00343+dt10_38,
#        't20_33':0.009/z,'t21_38':0.016/z,
#        't02_33':-0.045/z,'t10_49':0.311/z+0.0122+dt10_49,
#        't22_33':0.027/z,'t21_49':-0.019/z,
#        't11_34':0.099/z,'t10_410':0.18j/z,
#        't11_35':0.146j/z,
#        'E1':0.014/z+dxy,'E2':-0.539/z+0.02994+dXY,'E3':0.020/z+0.0,'E4':0.02/z+0.0,'E5':-0.581/z+0.03227+dz2,
#        't101_16':0.00275,
#        't001_11':0,'t121_16':-0.017/z,
#        't111_11':0,'t101_18':0.009j/z,
#        't201_11':0.017/z-0.00275,'t101_19':0.02j/z,
#        't201_14':0.003j/z+0.0045j,'t211_19':-0.0031j/z,
#        't001_33':0.011/z,'t101_38':0.006/z+0.00253+dt101_38,
#        't201_33':-0.008/z,'t121_38':-0.003/z,
#        't021_33':0.020/z,'t101_39':0.015/z,
#        't101_49':0.025/z-0.00203+dt101_49,'t121_49':0.006/z
#        }


kz_renorm = 1.0
kz_21 =1.0
kz_11 = 1.0
kz_31 = 1.0
kz_01 = 1.0
kz2 = 1.0



        
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
#Hpm3[1] = "1,8,-4.0*t101_18*sin(kx)*sin(kz),-4*t101_19*sin(ky)*sin(kz),2.0j*t211_19*sin(k1y)*exp(+kz),-2.0j*t211_19*sin(k2y)*exp(-kz)"
#Hpm3[2] = "1,9,-4.0*t101_18*sin(ky)*sin(kz),-4*t101_19*sin(kx)*sin(kz),2.0j*t211_19*sin(k1x)*exp(+kz),2.0j*t211_19*sin(k2x)*exp(-kz)"
Hpm3[3] = "3,8,4*t101_38*cos(kx)*cos(kz),4*t101_49*cos(ky)*cos(kz),2*t121_38*cos(k1x)*exp(+kz),2*t121_38*cos(k2x)*exp(-kz),2*t121_49*cos(k1y)*exp(+kz),2*t121_49*cos(k2y)*exp(-kz)"
Hpm3[4] = "3,9,4.0j*t101_39*cos(kx)*sin(kz),4.0j*t101_39*cos(ky)*sin(kz)" 
Hpm3[5] = "4,9,4*t101_49*cos(kx)*cos(kz),4*t101_38*cos(ky)*cos(kz),2*t121_49*cos(k1x)*exp(+kz),2*t121_49*cos(k2x)*exp(-kz),2*t121_38*cos(k1y)*exp(+kz),2*t121_38*cos(k2y)*exp(-kz)"


#Hpm3[1] = "1,8,-4.0*t101_18*sin(kx)*cos(kz),-4.0*t101_19*sin(ky)*sin(kz),2.0j*t211_19*sin(k1y)*exp(+kz),-2.0j*t211_19*sin(k2y)*exp(-kz)"
#Hpm3[2] = "1,9,-4.0*t101_18*sin(ky)*cos(kz),-4.0*t101_19*sin(kx)*sin(kz),2.0j*t211_19*sin(k1x)*exp(+kz),2.0j*t211_19*sin(k2x)*exp(-kz)"

basis = ["3dxy","3dXY","1.0j*3dxz","1.0j*3dyz","3dZR","3dxy","3dXY","-1.0j*3dxz","-1.0j*3dyz","3dZR"]
S = 1.0# for LiFeAs


def transform(Helements):
    tmp = []
    inds = np.array([1.0,1.0,-1.0j,-1.0j,1.0,1.0,1.0,1.0j,1.0j,1.0],dtype=complex)

    for h in Helements:

        hprime =np.conj(inds[int(h[0])])*complex(h[-1])*inds[int(h[1])]    
        if abs(np.imag(hprime))>0:
            print('Error in Hamiltonian--complex value found for REAL orbital basis')
        new_val = [*h[:5],str(np.real(hprime))]
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

    
def writefile(H,filename,mode):
    with open(filename,mode) as tofile:
        for i in range(len(H)):
            wl = ""
            for j in H[i]:
                wl += j +","
            wl +="\n"
            tofile.write(wl)
        tofile.close
        
        
def make_Ham():
    

    Hlist = fillH(Hpp,Hpm,tdic)  
    Hlist3D = fillH(Hpp3,Hpm3,tdic)
    Htrans_d=transform(Hlist)
    H3Dtrans_d=transform(Hlist3D) #convert to real orbitals since this model uses idxz and idyz

    filename = "FeSe_BMA_MOD.txt"
    

    writefile(Htrans_d,filename,"w")
    writefile(H3Dtrans_d,filename,"a")

    print('complete')
    return filename
    
    
if __name__ == "__main__":


    filename = make_Ham()




