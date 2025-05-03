# 独立绘图线程

# 对前向仿真结果的绘图
def PLOTOBJ_ploting(PLOTOBJ,CCD_simu,sup,path,name,plt,os):
    N=max(PLOTOBJ.shape)
    plt.figure(figsize=(10, 5))
    plt.subplot(2,4,1),plt.imshow(abs(PLOTOBJ [int(N/4):int(3*N/4),int(N/4):int(3*N/4),0]), cmap='hot'),plt.title('Focus on Plane1',family='Times New Roman',fontsize=12),plt.axis('off')
    plt.subplot(2,4,2),plt.imshow(abs(PLOTOBJ [int(N/4):int(3*N/4),int(N/4):int(3*N/4),1]), cmap='hot'),plt.title('Focus on Plane2',family='Times New Roman',fontsize=12),plt.axis('off')
    plt.subplot(2,4,3),plt.imshow(abs(PLOTOBJ [int(N/4):int(3*N/4),int(N/4):int(3*N/4),2]), cmap='hot'),plt.title('Focus on Plane3',family='Times New Roman',fontsize=12),plt.axis('off')
    plt.subplot(1,4,4),plt.imshow(abs(CCD_simu[int(N/4):int(3*N/4),int(N/4):int(3*N/4)  ]), cmap='hot'),plt.title('Simulated  CCD' ,family='Times New Roman',fontsize=12),plt.axis('off')
    plt.subplot(2,4,5),plt.imshow(abs(sup     [int(N/4):int(3*N/4),int(N/4):int(3*N/4),0]), cmap='hot'),plt.title('SUP  on  Plane1',family='Times New Roman',fontsize=12),plt.axis('off')
    plt.subplot(2,4,6),plt.imshow(abs(sup     [int(N/4):int(3*N/4),int(N/4):int(3*N/4),1]), cmap='hot'),plt.title('SUP  on  Plane2',family='Times New Roman',fontsize=12),plt.axis('off')
    plt.subplot(2,4,7),plt.imshow(abs(sup     [int(N/4):int(3*N/4),int(N/4):int(3*N/4),2]), cmap='hot'),plt.title('SUP  on  Plane3',family='Times New Roman',fontsize=12),plt.axis('off')
    filename = os.path.join(path, str(name) + '.png')
    plt.savefig(filename,dpi=1200)
    plt.show(block=False)
    plt.pause(5)
    plt.close()
    return '前向仿真结果保存在：'+filename

# 对重建恢复的绘图
def Result_ploting(RE_obj,RE_sup,Sum_sup,path,name,plt,os):
    N=max(RE_obj.shape)
    plt.figure(figsize=(10, 5))
    plt.subplot(2,4,1),plt.imshow(abs(RE_obj [int(N/4):int(3*N/4),int(N/4):int(3*N/4),0]), cmap='hot'),plt.title('OBJ1'       ,family='Times New Roman',fontsize=12),plt.axis('off')
    plt.subplot(2,4,2),plt.imshow(abs(RE_obj [int(N/4):int(3*N/4),int(N/4):int(3*N/4),1]), cmap='hot'),plt.title('OBJ2'       ,family='Times New Roman',fontsize=12),plt.axis('off')
    plt.subplot(2,4,3),plt.imshow(abs(RE_obj [int(N/4):int(3*N/4),int(N/4):int(3*N/4),2]), cmap='hot'),plt.title('OBJ3'       ,family='Times New Roman',fontsize=12),plt.axis('off')
    plt.subplot(1,4,4),plt.plot  (abs(Sum_sup)                                                       ),plt.title('Similarity' ,family='Times New Roman',fontsize=12)
    plt.subplot(2,4,5),plt.imshow(abs(RE_sup [int(N/4):int(3*N/4),int(N/4):int(3*N/4),0]), cmap='hot'),plt.title('SUP  1'     ,family='Times New Roman',fontsize=12),plt.axis('off')
    plt.subplot(2,4,6),plt.imshow(abs(RE_sup [int(N/4):int(3*N/4),int(N/4):int(3*N/4),1]), cmap='hot'),plt.title('SUP  2'     ,family='Times New Roman',fontsize=12),plt.axis('off')
    plt.subplot(2,4,7),plt.imshow(abs(RE_sup [int(N/4):int(3*N/4),int(N/4):int(3*N/4),2]), cmap='hot'),plt.title('SUP  3'     ,family='Times New Roman',fontsize=12),plt.axis('off')
    filename = os.path.join(path, str(name) + '.png')
    plt.savefig(filename,dpi=1200)
    plt.show(block=False)
    plt.pause(5)
    plt.close()
    return '迭代重建结果保存在：'+filename


















