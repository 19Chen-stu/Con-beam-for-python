# 正则化图像去噪，引用螺旋坐标
# Pic_input:输入图像
# AMP:正则化强度
# time:迭代次数 
# scope:裁剪范围
# target_move:聚焦移动范围
# CPU&GPU
#cuda>torch>np.int>np.float32:测试用时
def TV(Pic_input,AMP,time,scope,target_move,np,tqdm):
    #获取数据初始类型，因为在这段脚本中精度不需要太高，降低计算精度以节约时间
    original_type=Pic_input.dtype
    original_type_str = str(original_type)
    type_name = original_type_str.split('.')[0]
    if type_name=='torch':
        Pic_input=Pic_input.to(np.float32)
        Pic_shape=Pic_input.shape
    else:
        Pic_input=Pic_input.astype(np.float32)
        Pic_shape=np.shape(Pic_input)
    # 获取图像大小，新建输出图像
    # Pic_shape=np.shape(Pic_input)   
    Pic_Output=Pic_input
    # 初始化变分能量
    energy=0
    # 裁剪图像
    Pic_input=Pic_input[int((Pic_shape[0]-scope[0]+target_move[0])/2)-1:int((Pic_shape[0]-scope[0]+target_move[0])/2)+scope[0]-1,int((Pic_shape[1]-scope[1]+target_move[1])/2)-1:int((Pic_shape[1]-scope[1]+target_move[1])/2)+scope[1]-1]
    Pic_output=Pic_input
    Pic_input1=Pic_output
    # 螺旋索引生成
    [_,locationXY1]=spiral_index(scope[0],scope[1],1,np,tqdm)
    [_,locationXY2]=spiral_index(scope[0],scope[1],2,np,tqdm)
    #初始螺旋
    locationXY=locationXY1  
    #迭代首尾位置
    Top=2*(sum(scope))-4
    End=scope[0]*scope[1]-1
    # 迭代
    for i in range(time):
        for j in range(Top,End+1):
            fx=(Pic_output[locationXY[j,0]+1,locationXY[j,1]  ]-Pic_output[locationXY[j,0],locationXY[j,1]  ])
            fy=(Pic_output[locationXY[j,0]  ,locationXY[j,1]+1]-Pic_output[locationXY[j,0],locationXY[j,1]-1])/2
            grad=np.sqrt(fx**2+fy**2)
            if grad != 0:
                co1=1/grad
            else:
                co1=grad

            fx=(Pic_output[locationXY[j,0]  ,locationXY[j,1]  ]-Pic_output[locationXY[j,0]-1,locationXY[j,1]  ])
            fy=(Pic_output[locationXY[j,0]-1,locationXY[j,1]+1]-Pic_output[locationXY[j,0]-1,locationXY[j,1]-1])/2
            grad=np.sqrt(fx**2+fy**2)
            if grad != 0:
                co2=1/grad
            else:
                co2=grad

            fx=(Pic_output[locationXY[j,0]+1,locationXY[j,1]  ]-Pic_output[locationXY[j,0]-1,locationXY[j,1]])/2
            fy=(Pic_output[locationXY[j,0]  ,locationXY[j,1]+1]-Pic_output[locationXY[j,0]  ,locationXY[j,1]])
            grad=np.sqrt(fx**2+fy**2)
            if grad != 0:
                co3=1/grad
            else:
                co3=grad

            fx=(Pic_output[locationXY[j,0]+1,locationXY[j,1]-1]-Pic_output[locationXY[j,0]-1,locationXY[j,1]-1])/2
            fy=(Pic_output[locationXY[j,0]  ,locationXY[j,1]  ]-Pic_output[locationXY[j,0]  ,locationXY[j,1]-1])
            grad=np.sqrt(fx**2+fy**2)
            if grad != 0:
                co4=1/grad
            else:
                co4=grad

            Pic_output[locationXY[j,0],locationXY[j,1]]=(Pic_input[locationXY[j,0],locationXY[j,1]]+(1/AMP)*(co1*Pic_output[locationXY[j,0]+1,locationXY[j,1]]+co2*Pic_output[locationXY[j,0]-1,locationXY[j,1]]+co3*Pic_output[locationXY[j,0],locationXY[j,1]+1]+co4*Pic_output[locationXY[j,0],locationXY[j,1]-1]))*(1/(1+(1/(AMP)*(co1+co2+co3+co4))))

    Pic_output[1:-1,0]=Pic_output[1:-1,1]
    Pic_output[1:-1,-1]=Pic_output[1:-1,-2]
    Pic_output[0,1:-1]=Pic_output[1,1:-1]
    Pic_output[-1,1:-1]=Pic_output[-2,1:-1]

    Pic_output[0,0]=Pic_output[1,1]
    Pic_output[-1,-1]=Pic_output[-2,-2]    
    Pic_output[0,-1]=Pic_output[1,-2]
    Pic_output[-1,0]=Pic_output[-2,1]

    if (i+1)%2==0:
        locationXY=locationXY2
        locationXY2=locationXY1
        locationXY1=locationXY

    energy=np.sum(np.sqrt(np.diff(Pic_output[1:,1:-1],axis=0)**2+np.diff(Pic_output[1:-1,1:],axis=1)**2)+AMP*(Pic_output[1:-1,1:-1]-Pic_input1[1:-1,1:-1])**2)
    Pic_Output[int((Pic_shape[0]-scope[0]+target_move[0])/2)-1:int((Pic_shape[0]-scope[0]+target_move[0])/2)+scope[0]-1,int((Pic_shape[1]-scope[1]+target_move[1])/2)-1:int((Pic_shape[1]-scope[1]+target_move[1])/2)+scope[1]-1]=Pic_output

    # 数据类型回滚
    if type_name=='torch':
        Pic_Output=Pic_Output.to(original_type)
    else:
        Pic_Output=Pic_Output.astype(original_type)

    return Pic_Output,energy

# 螺旋索引生成函数
def spiral_index(X,Y,type,np,tqdm):
    location=np.ones([X,Y],dtype=int)*(X*Y+1)
    locationXY=np.zeros([X*Y,2],dtype=int)
    # 方向向量
    if str(np).split('f')[:-1]==["<module 'torch' "]:
        direction1=np.tensor([[0,1],[1,0],[0,-1],[-1,0]],dtype=np.int32)
        direction2=np.tensor([[1,0],[0,1],[-1,0],[0,-1]],dtype=np.int32)
        # direction1=t.LongTensor([[0,1],[1,0],[0,-1],[-1,0]])
        # direction2=t.LongTensor([[1,0],[0,1],[-1,0],[0,-1]])
    else:
        direction1=np.array([[0,1],[1,0],[0,-1],[-1,0]])
        direction2=np.array([[1,0],[0,1],[-1,0],[0,-1]])
    # 装填方向矩阵
    direction=[]
    # 初始位置
    locationX=0
    locationY=0
    # 起始方向
    direction_Now=0
    # 旋转方向
    if type==1:
        direction=direction1
        print('螺旋方向：顺时针旋转。开始生成螺旋索引')
    elif type==2:
        direction=direction2
        print('螺旋方向：逆时针旋转。开始生成螺旋索引')
    else:
        print('Error: 类型必须为顺时针或逆时针')
    with tqdm(total=X*Y) as pbar:
        for i in range(X*Y):
            location[locationX,locationY]=i

            locationXY[i,0]=locationX
            locationXY[i,1]=locationY

            locationNextX=locationX+direction[direction_Now,0]
            locationNextY=locationY+direction[direction_Now,1]
            if (locationNextX<X and locationNextX>=0) and (locationNextY<Y and locationNextY>=0) and (location[locationNextX,locationNextY]==(X*Y+1)):
                locationX=locationNextX
                locationY=locationNextY
            else:
                direction_Now=(direction_Now+1)%4
                locationX=locationX+direction[direction_Now,0]
                locationY=locationY+direction[direction_Now,1]
            pbar.update(1)
    print('完成生成螺旋索引')
    return location,locationXY

# 以下是调试代码
# import time
# import numpy as np
# import torch as t
# from tqdm import tqdm
# import scipy.io as sio
# data=sio.loadmat(r'E:\python代码\会聚球面光束仿真py重建\matlab.mat')

# a=time.time()

# matrix=np.array(data['a'],dtype=np.int8) #当输入是inn8时
# b,_=TV(matrix,1,1,[8,9],[0,0],np,tqdm) 
# # print(b)
# b=time.time()-a
# print(b)

# a=time.time()
# matrix=np.array(data['a'],dtype=np.float32) #当输入是浮点数时
# b,_=TV(matrix,1,1,[8,9],[0,0],np,tqdm) 
# # print(b)
# b=time.time()-a
# print(b)

# a=time.time()
# matrix=np.array(data['a'],dtype=np.float32)   #当输入是torch的浮点数时
# matrix=t.tensor(matrix,dtype=t.float64)
# b,_=TV(matrix,1,1,[8,9],[0,0],t,tqdm) 
# # print(b)
# b=time.time()-a 
# print(b)

# a=time.time() 
# matrix=np.array(data['a'],dtype=np.float32)   #当输入是torch的浮点数时,cuda测试
# matrix=t.tensor(matrix,dtype=t.float64).to(device='cuda:0')
# b,_=TV(matrix,1,1,[8,9],[0,0],t,tqdm) 
# # print(b)
# b=time.time()-a 
# print(b)