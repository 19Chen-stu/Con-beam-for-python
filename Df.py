# TV正则化+迭代边缘软阈值稀疏化
def Df(RE_Pic,SUP,Type,num1,num2,value1,value2,TV_C,np,tqdm,TVr):
   

    if Type==1:
        value=np.mean(np.abs(RE_Pic[SUP]))
        print("边缘稀疏化")
        mask=(np.abs(RE_Pic)<(value1*value)) & SUP
        Point=np.where(mask==True)
        Point1=np.ravel_multi_index(Point,RE_Pic.shape)  # 按行优先
        SUP[np.unravel_index(Point1[0:num1[0]], SUP.shape)]=False
        SUP[np.unravel_index(Point1[-num1[1]:], SUP.shape)]=False
        
        SUP=np.rot90(SUP,k=1)
        Point=np.where(mask==True)
        Point1=np.ravel_multi_index(Point,RE_Pic.shape)  # 按列优先
        SUP[np.unravel_index(Point1[0:num2[0]], SUP.shape)]=False
        SUP[np.unravel_index(Point1[-num2[1]:], SUP.shape)]=False

        SUP=np.rot90(SUP,k=3)

        # 螺旋TV正则化
    elif Type==2:
        [RE_Pic,_]=TVr.TV(np.abs(RE_Pic),TV_C.AMP,TV_C.time,TV_C.scope,TV_C.target_move,np,tqdm)
        value=np.mean(np.abs(RE_Pic[SUP]))
        SUP=np.abs(RE_Pic)>=(value2*value)
        # RE_Pic=(np.max(np.abs(RE_Pic[SUP]))-np.min(np.abs(RE_Pic[SUP])))/2*np.where(SUP,np.ones(RE_Pic.shape),0)
        

    return SUP,value,RE_Pic

#以下是测试代码
# import numpy as np
# import TV_regularization as TVr
# from tqdm import tqdm

# class TV_parameters:
#     def __init__(self,AMP,time,scope,target_move):
#         self.AMP=AMP
#         self.time=time
#         self.scope=scope.astype(np.int16)
#         self.target_move=target_move.astype(np.int16)
       
    
#     def __repr__(self):
#         return f"A(AMP={self.AMP},time={self.time},scope={self.scope},target_move={self.target_move})"
    
# TV1_C=TV_parameters(AMP=1,time=2,scope=np.array([4,3]),target_move=np.array([0,0]))

# a=np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20],[21,22,23,24,25],[26,27,28,29,30]])
# sup=np.array([[0,0,0,0,0],[0,1,1,1,0],[0,1,1,1,0],[0,1,1,1,0],[0,1,1,1,0],[0,1,1,1,0]]).astype(bool)



# SUP,_=Df(a,sup,2,[2,2],[2,2],0.1,0.1,TV1_C,np,tqdm,TVr)







