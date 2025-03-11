"""
word文档中对应的大部件相似度对比：
"""
import sys
import os
# 获取当前文件的绝对路径
current_path = os.path.dirname(os.path.abspath(__file__))
# 获取上一级目录的路径
parent_path = os.path.dirname(current_path)
# 将上一级目录添加到 sys.path
sys.path.append(parent_path)

from PIL import ImageDraw, ImageFont,Image
import numpy as np
import xml.etree.ElementTree as ET


if __name__ == "__main__":
    #img_path = r"E:\N-T2-20230531-CRH380AL2621-07-A-02-L-SSQZ-01-04.jpg"
    import xml.etree.ElementTree as ET
    #from utils import read_xml_std_boxs
    import os
    import time
    from Sim_3D import Sim_3D
    start_time = time.time()
    sim=Sim_3D()

    # 异常1   N-T1-20231213-CRH380AL2611-01-A-01-0-DBZC-01-09.jpg
    # 正常1   N-T1-20231213-CRH380AL2611-01-B-01-0-DBZC-01-17.jpg
    # 正常2   N-T1-20231213-CRH380AL2611-01-B-01-0-DBZC-01-09.jpg
    a =   r"D:\adavance\Similarity_comparison_3D\test_data\test_big\N-T1-20231213-CRH380AL2611-01-A-01-0-DBZC-01-09.jpg"#异常
    b =   r"D:\adavance\Similarity_comparison_3D\test_data\test_big\N-T1-20231213-CRH380AL2611-01-B-01-0-DBZC-01-09.jpg"#正常
    c =   r'D:\adavance\Similarity_comparison_3D\test_data\test_big\N-T1-20231213-CRH380AL2611-01-B-01-0-DBZC-01-17.jpg'#正常

    norm_file=b    #此变量最好放正常图像的路径,标准图像
    abnorm_file=a  #此变量待检测图像路径，可以是正常的，也可以是异常的
    norm_file=r"D:\adavance\Pyqt-main\DetectionModules\Algo1\Similarity_comparison_3D\test_data\daijian\clxz2.jpg"    #此变量最好放正常图像的路径,标准图像
    abnorm_file=r"D:\adavance\Pyqt-main\DetectionModules\Algo1\Similarity_comparison_3D\test_data\daijian\clxz1.jpg" #此变量待检测图像路径，可以是正常的，也可以是异常的

    abnormbox=[0,0,1920-0,1200-0,0,0]#取图像中一个非常大的框作为目标框 10,10,1920-10,1200-10
    normbox=abnormbox

    Sim_3D_dis_thresh=10#距离阈值
    Sampling_interval=16 #点云降采样系数
    result_compare,icp_error=sim.abnorm_detect_3D(norm_file, normbox, abnorm_file, abnormbox, Sim_3D_dis_thresh,Sampling_interval)
    if result_compare==0:
        print('该区域点云无效')
    elif result_compare==1:
        print('点云平均距离相差较小，可认为大概率是正常的部件')
    elif result_compare==2:
        print('点云平均距离相差较大，可认为大概率有缺陷的部件')
    
    all_time = time.time() - start_time
    print(f'总用时{all_time}s')


          



  

