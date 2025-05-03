#按照等边三角形的方式排列目标，并返回目标对象、支持、相对支撑
def target_make(target_size,target_path,sup_scope,ps,N,np,cv2):
    # target_size：目标参数
    # target_path：目标路径
    # sup_scope：物体支撑留空距离
    # ps：目标尺寸
    # N：目标场大小
    # np：引入声明库函数
    sup_scope=np.array(sup_scope)
    Obj=np.zeros([N,N,3])
    sup=np.zeros((N,N,3),dtype=bool)
    supp=np.ones((N,N,3),dtype=bool)

    #设置目标物体的预定大小
    target_size=np.fix(target_size/ps)
    
    #通过opencv读入图片并矩阵化,预处理,不考虑图像是不是彩色的，一律按照灰度图像处理
    image0=cv2.imdecode(np.fromfile(target_path[0],dtype=np.uint8),-1)
    image0=cv2.resize(image0,(np.int8(target_size[0]),np.int8(target_size[0])))
    image0=np.array(image0)
    image1=cv2.imdecode(np.fromfile(target_path[1],dtype=np.uint8),-1)
    image1=cv2.resize(image1,(np.int8(target_size[1]),np.int8(target_size[1])))
    image1=np.array(image1)
    image2=cv2.imdecode(np.fromfile(target_path[2],dtype=np.uint8),-1)
    image2=cv2.resize(image2,(np.int8(target_size[2]),np.int8(target_size[2])))
    image2=np.array(image2)

    #判断图像是否灰度图像，图像灰度化
    if len(image0.shape)==3:
        image0=cv2.cvtColor(image0,cv2.COLOR_BGR2GRAY)
    if len(image1.shape)==3:
        image1=cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
    if len(image2.shape)==3:
        image2=cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)

    #设置阈值
    image0[image0<=220]=0
    image1[image1<=220]=0
    image2[image2<=220]=0

    #强度反转，镂空与阻挡反转
    image0=~image0
    image1=~image1
    image2=~image2
    image0=image0/255
    image1=image1/255
    image2=image2/255
    
    
   #填充图片到图像正中心
    Obj[int((N-target_size[0])/2):int((N-target_size[0])/2+target_size[0]),
        int((N-target_size[0])/2):int((N-target_size[0])/2+target_size[0])
        ,0]=image0
    Obj[int((N-target_size[1])/2):int((N-target_size[1])/2+target_size[1]),
        int((N-target_size[1])/2):int((N-target_size[1])/2+target_size[1])
        ,1]=image1
    Obj[int((N-target_size[2])/2):int((N-target_size[2])/2+target_size[2]),
        int((N-target_size[2])/2):int((N-target_size[2])/2+target_size[2])
        ,2]=image2

    #平移目标物体
    Obj[:,:,0]=np.roll(Obj[:,:,0], -int(1.5*target_size[0]), axis=0)

    Obj[:,:,1]=np.roll(Obj[:,:,1],  int(1.5*target_size[1]), axis=0)
    Obj[:,:,1]=np.roll(Obj[:,:,1], -int(1.5*target_size[1]), axis=1)

    Obj[:,:,2]=np.roll(Obj[:,:,2],  int(1.5*target_size[2]), axis=0)
    Obj[:,:,2]=np.roll(Obj[:,:,2],  int(1.5*target_size[2]), axis=1)

    #创建支撑
    sup[int((N-target_size[0]-sup_scope[0])/2):int((N-target_size[0]-sup_scope[0])/2+sup_scope[0]+target_size[0]),
        int((N-target_size[0]-sup_scope[1])/2):int((N-target_size[0]-sup_scope[1])/2+sup_scope[1]+target_size[0]),
        0]=1

    sup[int((N-target_size[1]-sup_scope[0])/2):int((N-target_size[1]-sup_scope[0])/2+sup_scope[0]+target_size[1]),
        int((N-target_size[1]-sup_scope[1])/2):int((N-target_size[1]-sup_scope[1])/2+sup_scope[1]+target_size[1]),
        1]=1

    sup[int((N-target_size[2]-sup_scope[0])/2):int((N-target_size[2]-sup_scope[0])/2+sup_scope[0]+target_size[2]),
        int((N-target_size[2]-sup_scope[1])/2):int((N-target_size[2]-sup_scope[1])/2+sup_scope[1]+target_size[2]),
        2]=1

    #移动支撑
    sup[:,:,0]=np.roll(sup[:,:,0], -int(1.5*target_size[0]), axis=0)

    sup[:,:,1]=np.roll(sup[:,:,1],  int(1.5*target_size[1]), axis=0)
    sup[:,:,1]=np.roll(sup[:,:,1], -int(1.5*target_size[1]), axis=1)

    sup[:,:,2]=np.roll(sup[:,:,2],  int(1.5*target_size[2]), axis=0)
    sup[:,:,2]=np.roll(sup[:,:,2],  int(1.5*target_size[2]), axis=1)

    supp[0:int(N/2+sup_scope[0]),:,0]=0
    supp[int(N/2-sup_scope[0]):N,0:int(N/2+sup_scope[1]),1]=0
    supp[int(N/2-sup_scope[0]):N,int(N/2-sup_scope[1]):N,2]=0






    return [Obj,sup,supp]