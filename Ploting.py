# 用于绘制不同深度的光束衍射图
def Ploting(UIN,OBJ,Z,fxx,fyy,Lambda,np,ad):
    
    PLOTOBJ=np.zeros(np.shape(OBJ),dtype=complex)

    Obj1=ad.Angular_diffraction(UIN,fxx,fyy,Z[0],Lambda,np)*OBJ[:,:,0]
    Obj2=ad.Angular_diffraction(UIN,fxx,fyy,Z[1],Lambda,np)*OBJ[:,:,1]
    Obj3=ad.Angular_diffraction(UIN,fxx,fyy,Z[2],Lambda,np)*OBJ[:,:,2]

    PLOTOBJ[:,:,0]=(ad.Angular_diffraction(Obj1,fxx,fyy,Z[0]-Z[0],Lambda,np)+
        ad.Angular_diffraction(Obj2,fxx,fyy,Z[0]-Z[1],Lambda,np)+
            ad.Angular_diffraction(Obj3,fxx,fyy,Z[0]-Z[2],Lambda,np))
        
    PLOTOBJ[:,:,1]=(ad.Angular_diffraction(Obj1,fxx,fyy,Z[1]-Z[0],Lambda,np)+
        ad.Angular_diffraction(Obj2,fxx,fyy,Z[1]-Z[1],Lambda,np)+
            ad.Angular_diffraction(Obj3,fxx,fyy,Z[1]-Z[2],Lambda,np))
    
    PLOTOBJ[:,:,2]=(ad.Angular_diffraction(Obj1,fxx,fyy,Z[2]-Z[0],Lambda,np)+
        ad.Angular_diffraction(Obj2,fxx,fyy,Z[2]-Z[1],Lambda,np)+
            ad.Angular_diffraction(Obj3,fxx,fyy,Z[2]-Z[2],Lambda,np))

    PLOTOBJ=PLOTOBJ*np.conj(PLOTOBJ)


    return PLOTOBJ