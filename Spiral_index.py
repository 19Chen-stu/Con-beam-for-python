#给定矩阵大小参数，生成螺旋索引
# import numpy as np
# from tqdm import tqdm
# 螺旋索引生成函数
# numpy、torch都可以

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

