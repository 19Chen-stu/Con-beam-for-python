#python重写ABC三目标会聚球面光束照明仿真程序
#三个物体ABC仿真多深度最终版
# CCD裁剪生成频谱中心区域
# 可调节不同采样精度
# from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch as t
import sys
import os
from tqdm import tqdm
import target_make  as Tm    #导入自定义包
import Show as SH         #导入自定义包
import Ploting as PLT       #导入自定义包
import Df as Df           #导入自定义包
import TV_regularization as TVr   #导入自定义包


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.append(os.path.abspath("D:\python代码\角谱衍射自建"))    #添加自定义包的路径
import Ang_diffraction as ad    # type: ignore #导入自定义包



# 定义参数
Lambda=532e-9              #波长
f=200e-3                  #透镜焦距
K=2*np.pi/Lambda             #波数
N=1024
Long=11e-3
use_GPU=1
save_path='D:\python代码\会聚球面光束仿真py重建\RESULT'   #结果保存路径

target_size=np.array([0.6e-3,0.6e-3,0.6e-3])   #单个目标尺寸
target_path=["D:\python代码\会聚球面光束仿真py重建\字母A.png",
             "D:\python代码\会聚球面光束仿真py重建\字母B.png",
             "D:\python代码\会聚球面光束仿真py重建\字母C.png"]   #目标文件地址

(xx,yy,xita,r,fxx,fyy) = ad.C_parameter(N,Long,np)
(LenPhase)=ad.Lens(xx,yy,f,Lambda,0,0,np)
(Obj,sup,_)=Tm.target_make(target_size,target_path,[10,10],Long/N,N,np,cv2)

z=np.array([0,5,10])*1e-3  #距离
noise_energy=np.float64([1e-4])

# 定义前向传播模型
Uin=LenPhase
Uout=ad.Angular_diffraction(Uin,fxx,fyy,z[0],Lambda,np)*Obj[:,:,0]
Uout0=ad.Angular_diffraction(Uout,fxx,fyy,f-z[0],Lambda,np)

Uout=ad.Angular_diffraction(Uin,fxx,fyy,z[1],Lambda,np)*Obj[:,:,1]
Uout1=ad.Angular_diffraction(Uout,fxx,fyy,f-z[1],Lambda,np)

Uout=ad.Angular_diffraction(Uin,fxx,fyy,z[2],Lambda,np)*Obj[:,:,2]
Uout2=ad.Angular_diffraction(Uout,fxx,fyy,f-z[2],Lambda,np)

Uout=Uout0+Uout1+Uout2
PLOTOBJ=PLT.Ploting(Uin,Obj,z,fxx,fyy,Lambda,np,ad)
# 处理噪声
CCD_simu=Uout*np.conj(Uout)
CCD_simu=CCD_simu/np.max(CCD_simu)
CCD_simu=np.uint16(np.abs(CCD_simu)*65535)
CCD_simu=np.float64(CCD_simu)
Random_noise=np.random.uniform(0,1,size=CCD_simu.shape)*noise_energy*65535
CCD_simu=np.sqrt(CCD_simu+Random_noise)
CCD_simu=CCD_simu/np.max(CCD_simu)

# 设置一部分绘图结果
note=SH.PLOTOBJ_ploting(PLOTOBJ,CCD_simu,sup,save_path,'ABC',plt,os)
print(note)

# 定义TV参数
class TV_parameters:
    def __init__(self,AMP,time,scope,target_move):
        self.AMP=AMP
        self.time=time
        self.scope=scope.astype(np.int16)
        self.target_move=target_move.astype(np.int16)
       
    
    def __repr__(self):
        return f"A(AMP={self.AMP},time={self.time},scope={self.scope},target_move={self.target_move})"
    

TV_C1=TV_parameters(1e-3,100,np.array([50,50]),np.array([0,0]))
TV_C2=TV_parameters(1e-3,100,np.array([50,50]),np.array([0,0]))
TV_C3=TV_parameters(1e-3,100,np.array([50,50]),np.array([0,0]))
print(TV_C1)
print(TV_C1)
print(TV_C1)




# 设置反向迭代参数
beita=np.array([0.5],dtype=float)
cut_off_value=0.25
time_total=101
time_HIO=100
time_ER=50
time_shrink=5

# 设置反向迭代数据类型
if t.cuda.is_available()&use_GPU==True:
    device=t.device("cuda")
    RE_obj=t.ones([N,N,3],dtype=complex).to(device)
    BG1=t.zeros([N,N],dtype=complex).to(device)
    BG2=t.zeros([N,N],dtype=complex).to(device)
    BG3=t.zeros([N,N],dtype=complex).to(device)
    RE_forward1=t.zeros([N,N],dtype=complex).to(device)
    RE_forward2=t.zeros([N,N],dtype=complex).to(device)
    RE_forward3=t.zeros([N,N],dtype=complex).to(device)
    RE_back1=t.zeros([N,N],dtype=complex).to(device)
    RE_back2=t.zeros([N,N],dtype=complex).to(device)
    RE_back3=t.zeros([N,N],dtype=complex).to(device)
    CCD_RE=t.zeros([N,N],dtype=complex).to(device)
    theta=t.zeros([N,N],dtype=complex).to(device)
    Sum_sup=t.zeros([time_total+1]).to(device)
    CCD_simu=t.tensor(CCD_simu).to(device)
    fxx=t.tensor(fxx).to(device)
    fyy=t.tensor(fyy).to(device)    
    Lambda=t.tensor(Lambda).to(device)
    LenPhase=t.tensor(LenPhase).to(device)
    sup=t.tensor(sup).to(device)
    z=t.tensor(z).to(device)
    f=t.tensor(f).to(device)
    beita=t.tensor(beita).to(device)
    R_mode=t
    print('GPU available:',t.cuda.is_available())
else:
    RE_obj=np.ones([N,N,3],dtype=complex)
    BG1=np.zeros([N,N],dtype=complex)
    BG2=np.zeros([N,N],dtype=complex)
    BG3=np.zeros([N,N],dtype=complex)
    RE_forward1=np.zeros([N,N],dtype=complex)
    RE_forward2=np.zeros([N,N],dtype=complex)
    RE_forward3=np.zeros([N,N],dtype=complex)
    RE_back1=np.zeros([N,N],dtype=complex)
    RE_back2=np.zeros([N,N],dtype=complex)
    RE_back3=np.zeros([N,N],dtype=complex)
    CCD_RE=np.zeros([N,N],dtype=complex)
    theta=np.zeros([N,N],dtype=complex)
    Sum_sup=np.zeros([time_total])
    R_mode=np
    print('CPU')

# 开始反向迭代
for i in range(1,time_total+1):
    with tqdm(total=time_HIO) as pbar:
        for j in range(0,time_HIO):
            BG1=ad.Angular_diffraction(LenPhase,fxx,fyy,z[0],Lambda,R_mode)
            RE_forward1=ad.Angular_diffraction(BG1*RE_obj[:,:,0],fxx,fyy,f-z[0],Lambda,R_mode)

            BG2=ad.Angular_diffraction(LenPhase,fxx,fyy,z[1],Lambda,R_mode)
            RE_forward2=ad.Angular_diffraction(BG2*RE_obj[:,:,1],fxx,fyy,f-z[1],Lambda,R_mode)

            BG3=ad.Angular_diffraction(LenPhase,fxx,fyy,z[2],Lambda,R_mode)
            RE_forward3=ad.Angular_diffraction(BG3*RE_obj[:,:,2],fxx,fyy,f-z[2],Lambda,R_mode)

            CCD_RE=RE_forward1+RE_forward2+RE_forward3
            if j==0:
                Sum_sup[i-1]=R_mode.sum(R_mode.abs(R_mode.abs(CCD_RE)-CCD_simu))
            theta=R_mode.angle(CCD_RE)
            CCD_RE=CCD_simu*R_mode.exp(1j*theta)

            RE_back1=ad.Angular_diffraction(CCD_RE,fxx,fyy,-f+z[0],Lambda,R_mode)*R_mode.conj(BG1)            
            RE_back2=ad.Angular_diffraction(CCD_RE,fxx,fyy,-f+z[1],Lambda,R_mode)*R_mode.conj(BG2)
            RE_obj[:,:,1]=R_mode.where(sup[:,:,1],RE_back2,0+0j)+R_mode.where(~sup[:,:,1],(RE_obj[:,:,1]-beita*RE_back2),0+0j)

            RE_back3=ad.Angular_diffraction(CCD_RE,fxx,fyy,-f+z[2],Lambda,R_mode)*R_mode.conj(BG3)
            RE_obj[:,:,2]=R_mode.where(sup[:,:,2],RE_back3,0+0j)+R_mode.where(~sup[:,:,2],(RE_obj[:,:,2]-beita*RE_back3),0+0j)
            
            pbar.update(1)

    with tqdm(total=time_ER) as pbar:
        for k in range(0,time_ER):
            BG1=ad.Angular_diffraction(LenPhase,fxx,fyy,z[0],Lambda,R_mode)
            RE_forward1=ad.Angular_diffraction(BG1*RE_obj[:,:,0],fxx,fyy,f-z[0],Lambda,R_mode)

            BG2=ad.Angular_diffraction(LenPhase,fxx,fyy,z[1],Lambda,R_mode)
            RE_forward2=ad.Angular_diffraction(BG2*RE_obj[:,:,1],fxx,fyy,f-z[1],Lambda,R_mode)

            BG3=ad.Angular_diffraction(LenPhase,fxx,fyy,z[2],Lambda,R_mode)
            RE_forward3=ad.Angular_diffraction(BG3*RE_obj[:,:,2],fxx,fyy,f-z[2],Lambda,R_mode)

            CCD_RE=RE_forward1+RE_forward2+RE_forward3
            theta=R_mode.angle(CCD_RE)
            CCD_RE=CCD_simu*R_mode.exp(1j*theta)

            RE_back1=ad.Angular_diffraction(CCD_RE,fxx,fyy,-f+z[0],Lambda,R_mode)*R_mode.conj(BG1)
            RE_obj[:,:,0]=R_mode.where(sup[:,:,0],RE_back1, 0+0j)
            
            RE_back2=ad.Angular_diffraction(CCD_RE,fxx,fyy,-f+z[1],Lambda,R_mode)*R_mode.conj(BG2)
            RE_obj[:,:,1]=R_mode.where(sup[:,:,1],RE_back2,0+0j)

            RE_back3=ad.Angular_diffraction(CCD_RE,fxx,fyy,-f+z[2],Lambda,R_mode)*R_mode.conj(BG3)
            RE_obj[:,:,2]=R_mode.where(sup[:,:,2],RE_back3,0+0j)

            pbar.update(1)


    if np.mod(i,time_shrink)==0:
        # 不使用DF进行去噪正则化
        # if i<0.8*time_total:
        #     sup[:,:,0],_=Df.Df(R_mode.abs(RE_obj[:,:,0]),sup[:,:,0],1,([2,2]),([2,2]),cut_off_value,_,_,R_mode,_,_)
        #     sup[:,:,1],_=Df.Df(R_mode.abs(RE_obj[:,:,1]),sup[:,:,1],1,([2,2]),([2,2]),cut_off_value,_,_,R_mode,_,_)
        #     sup[:,:,2],_=Df.Df(R_mode.abs(RE_obj[:,:,2]),sup[:,:,2],1,([2,2]),([2,2]),cut_off_value,_,_,R_mode,_,_)
        # else:
        #     sup[:,:,0],_=Df.Df(R_mode.abs(RE_obj[:,:,0]),sup[:,:,0],2,_,_,_,cut_off_value,TV_C1,np,tqdm,TVr)
        #     sup[:,:,1],_=Df.Df(R_mode.abs(RE_obj[:,:,1]),sup[:,:,1],2,_,_,_,cut_off_value,TV_C2,np,tqdm,TVr)
        #     sup[:,:,2],_=Df.Df(R_mode.abs(RE_obj[:,:,2]),sup[:,:,2],2,_,_,_,cut_off_value,TV_C3,np,tqdm,TVr)
        BG1=R_mode.abs(RE_obj[:,:,0])
        sup[:,:,0]=BG1>cut_off_value*R_mode.mean(BG1[sup[:,:,0]])
        
        BG2=R_mode.abs(RE_obj[:,:,1])
        sup[:,:,1]=BG2>cut_off_value*R_mode.mean(BG2[sup[:,:,1]])

        BG3=R_mode.abs(RE_obj[:,:,2])
        sup[:,:,2]=BG3>cut_off_value*R_mode.mean(BG3[sup[:,:,2]])

        note=SH.Result_ploting(R_mode.abs(RE_obj).cpu().numpy(),sup.cpu().numpy(),Sum_sup.cpu().numpy(),save_path,'第'+str(i/5)+'次收缩',plt,os)
        print(note)

    print('第'+str(i)+'次反向迭代完成')


print('程序结束，结果保存在：'+save_path) 