import torch as t
import time 
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm
import Df as Df           #导入自定义包
import TV_regularization as TVr   #导入自定义包
sys.path.append(os.path.abspath("D:\python代码\角谱衍射自建"))    #添加自定义包的路径
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
class TV_parameters:
    def __init__(self,AMP,time,scope,target_move):
        self.AMP=AMP
        self.time=time
        self.scope=scope.astype(np.int32)
        self.target_move=target_move.astype(np.int32)
       
    
    def __repr__(self):
        return f"A(AMP={self.AMP},time={self.time},scope={self.scope},target_move={self.target_move})"
    

N=512
use_GPU=1
save_path='D:\python代码\会聚球面光束仿真py重建\TEST'   #结果保存路径

####读入图像
Pic_path = r'C:\Users\chen\Desktop\A.png'
img = cv.imdecode(np.fromfile(Pic_path, dtype=np.uint8), -1)
img=cv.resize(img,(np.int16(128),np.int16(128)))
img=np.array(img)
img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
img[img<=220]=0
img=(~img)
img=img/255
img=np.pad(img,((192,192),(192,192)))
plt.imshow(img,cmap='gray')
plt.show()
print('完成图像处理')
# 定义模拟CCD参数
Uout=np.abs(np.fft.fft2(img))
plt.imshow(np.abs(np.fft.fftshift(Uout)),cmap='gray')
filename = os.path.join(save_path, 'Uout.png')
plt.savefig(filename,dpi=1200)
plt.show(block=False)
plt.pause(1)
plt.close()

Uout=Uout+0.02*np.random.rand(N,N)*np.max(Uout)
sup=np.ones((200,200),dtype=bool)
sup=np.pad(sup,((156,156),(156,156)))   

# 定义参数
beita=np.array([0.5],dtype=float)
cut_off_value=0.55
time_total=101
time_HIO=400
time_ER=200
time_shrink=5
TV_C1=TV_parameters(2,2,np.array([200,200]),np.array([0,0]))

stat=time.time()

if t.cuda.is_available()&use_GPU==True:
    device=t.device("cuda")
    RE_obj=t.zeros([N,N],dtype=complex).to(device)    
    RE_forward=t.zeros([N,N],dtype=complex).to(device)       
    theta=t.zeros([N,N],dtype=complex).to(device)
    Uout=t.tensor(Uout).to(device)    
    sup=t.tensor(sup).to(device)    
    beita=t.tensor(beita).to(device)
    R_mode=t
    print('GPU available:',t.cuda.is_available())
else:
    RE_obj=np.zeros([N,N],dtype=complex)    
    RE_forward=np.zeros([N,N],dtype=complex)    
    CCD_RE=np.zeros([N,N],dtype=complex)
    theta=np.zeros([N,N],dtype=complex)
    R_mode=np
    print('CPU')



for i in range(1,time_total+1):
    with tqdm(total=time_HIO+time_ER) as pbar:
        for j in range(0,time_HIO):
            RE_forward=R_mode.fft.fft2(RE_obj)
            theta=R_mode.angle(RE_forward)
            CCD_RE=R_mode.exp(1j*theta)*Uout
            RE_forward=R_mode.fft.ifft2(CCD_RE)
            RE_obj=R_mode.where(sup,RE_forward,0+0j)+R_mode.where(~sup,(RE_obj-beita*RE_forward),0+0j)
            pbar.update(1)

        for k in range(0,time_ER):
            RE_forward=R_mode.fft.fft2(RE_obj)
            theta=R_mode.angle(RE_forward)
            CCD_RE=R_mode.exp(1j*theta)*Uout
            RE_forward=R_mode.fft.ifft2(CCD_RE)     
            RE_obj=R_mode.where(sup,RE_forward,0+0j)                    
            pbar.update(1)

    if np.mod(i,time_shrink)==0:
        if i<36:
            if use_GPU==True:
                SUP,_,OBJ=Df.Df(RE_obj.cpu().numpy(),sup.cpu().numpy(),1,[800,800],[800,800],np.array([0.6]),[],[],np,[],[])
                sup=R_mode.tensor(SUP).to(device)
                RE_obj=R_mode.tensor(OBJ).to(device)
            else:
                SUP,_,OBJ=Df.Df(RE_obj,sup,1,[800,800],[800,800],np.array([0.6]),[],TV_C1,np,tqdm,TVr)
                sup=SUP
                RE_obj=OBJ

        else:
            if use_GPU==True:
                SUP,_,OBJ=Df.Df(RE_obj.cpu().numpy(),sup.cpu().numpy(),2,[],[],[],cut_off_value,TV_C1,np,tqdm,TVr)
                sup=R_mode.tensor(SUP).to(device)
                RE_obj=R_mode.tensor(OBJ).to(device)
            else:
                SUP,_,OBJ=Df.Df(RE_obj,sup,2,[],[],[],cut_off_value,TV_C1,np,tqdm,TVr)
                sup=SUP
                RE_obj=OBJ

        plt.subplot(1,2,1)
        plt.imshow(np.abs(OBJ),cmap='gray')
        plt.subplot(1,2,2)
        plt.imshow(np.abs(SUP),cmap='gray')       
        filename = os.path.join(save_path, 'RE_obj'+str(i)+'.png')
        plt.savefig(filename,dpi=1200)
        plt.show(block=False)
        plt.pause(1)
        plt.close()
        print('第'+str(i)+'次迭代完成')     

plt.subplot(1,2,1)
plt.imshow(np.abs(RE_obj.cpu().numpy()),cmap='gray')
plt.subplot(1,2,2)
plt.imshow(np.abs(sup.cpu().numpy()),cmap='gray')       
filename = os.path.join(save_path, 'Result'+'.png')
plt.savefig(filename,dpi=1200)
plt.show(block=False)
plt.close()
print('结束')

end=time.time()
print('总耗时:',end-stat,'s')


















