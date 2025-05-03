#Ang_diffraction.py 是角谱的衍射传播子函数部分的python代码



def C_parameter(N,Long,np): #定义生成对应的空间空域和频域坐标，同时校准频域坐标的位置，额外输出对于的空域极坐标参数
    # N: 角谱的采样点数
    # Long: 角谱的长度
    # np: 引入声明库函数
    ps=Long/N
    x=ps*np.arange(-N/2,N/2)        # 空域坐标,范围左闭右开
    [xx,yy]=np.meshgrid(x,x)        # 空域坐标网格
    fx=1/Long*np.arange(-N/2,N/2)   # 频域坐标
    [fxx,fyy]=np.meshgrid(fx,fx)    # 频域坐标网格
    fxx=np.fft.ifftshift(fxx)       # 频域坐标校准
    fyy=np.fft.ifftshift(fyy)       # 频域坐标校准
    r=np.sqrt(xx**2+yy**2)          # 空域极径
    xita=np.arctan2(yy,xx)          # 空域极角
    
    return xx,yy,xita,r,fxx,fyy


def Lens(xx,yy,f,Lambda,Mod,N,np): #生成不同类型的透镜相位
    # xx,yy：空域坐标
    # f：焦距
    # Lambda：波长
    # Mod：类型控制参数
    # N：大小控制参数
    # np：引入声明库函数
    Len=np.exp(-1j*2*np.pi/Lambda/(2*f)*(xx**2+yy**2)) # 透镜相位
    if Mod==1: # 矩形透镜
        NA=(np.abs(xx)<np.max(xx)*N)&(np.abs(yy)<np.max(yy)*N)
        LenPhase=NA*Len
    elif Mod==2: # 圆形透镜
        NA=np.sqrt(xx**2+yy**2)<(np.max(xx)*N)
        LenPhase=NA*Len
    elif Mod==0: # 无透镜
         LenPhase=Len
    return LenPhase


def win(xx,yy,Mod,N,np): #生成不同类型窗口
    # xx,yy：空域坐标
    # Mod：类型控制参数
    # N：大小控制参数
    # np：引入声明库函数
    if Mod==1:     #矩形
        NA=(np.abs(xx)<(np.max(xx)*N))&(np.abs(yy)<(np.max(yy)*N))
    elif Mod==2:   #圆形
        NA=np.sqrt(xx**2+yy**2)<(np.min([np.max(yy),np.max(xx)])*N)
    elif Mod==3:   #十字沟道
        NA=(np.abs(xx)<np.max(xx)*N)|(np.abs(yy)<np.max(yy)*N)
    elif Mod==0:   #无透镜
        win=np.ones(np.shape(xx))
    return NA

def Angular_diffraction(Uin,fxx,fyy,z,Lambda,np):
    # Uin：输入光场
    # fxx,fyy：频域坐标
    # z：传播距离
    # Lambda：波长
    # np：引入声明库函数    
    H=np.exp(1j*2*np.pi*z*np.sqrt((1/Lambda)**2-fxx**2-fyy**2))
    Uout=np.fft.ifft2(np.fft.fft2(Uin)*H)
    return Uout