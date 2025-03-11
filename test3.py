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
import datetime
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
    save_path=   r"D:\adavance\Similarity_comparison_3D\test_data\result"   #输出图像所在的文件夹

    # 异常1   N-T1-20231213-CRH380AL2611-01-A-01-0-DBZC-01-09.jpg
    # 正常1   N-T1-20231213-CRH380AL2611-01-B-01-0-DBZC-01-17.jpg
    # 正常2   N-T1-20231213-CRH380AL2611-01-B-01-0-DBZC-01-09.jpg
    a =   r"D:\adavance\Similarity_comparison_3D\test_data\test_big\N-T1-20231213-CRH380AL2611-01-A-01-0-DBZC-01-09.jpg"#异常
    b =   r"D:\adavance\Similarity_comparison_3D\test_data\test_big\N-T1-20231213-CRH380AL2611-01-B-01-0-DBZC-01-09.jpg"#正常
    c =   r'D:\adavance\Similarity_comparison_3D\test_data\test_big\N-T1-20231213-CRH380AL2611-01-B-01-0-DBZC-01-17.jpg'#正常

    file='2.jpg'

    norm_file=  r"D:\adavance\Pyqt-main\DetectionModules\Algo1\Similarity_comparison_3D\test_data\daijian\wdbzc2.jpg"    #此变量最好放正常图像的路径,标准图像
    abnorm_file=r"D:\adavance\Pyqt-main\DetectionModules\Algo1\Similarity_comparison_3D\test_data\daijian\wdbzc2.jpg" #此变量待检测图像路径，可以是正常的，也可以是异常的

    # norm_file=  b
    # abnorm_file=a

    Sampling_interval=16 #点云降采样系数
    result_box=sim.deform_detect_3D(norm_file, abnorm_file, Sampling_interval)
    print(result_box)

    #绘制结果图像保存到文件夹
    abnormimage = Image.open(abnorm_file)
    print(abnormimage.size)
    draw = ImageDraw.Draw(abnormimage)
    text = f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    draw.text((30,30), text, fill="red", width=10,font=ImageFont.truetype("arial.ttf", 40))

    for box in result_box:
        draw.rectangle([box[0], box[1], box[2], box[3]], outline="blue", width=2)
        text = f"result: {box[4]}\ndistance: {box[5]}"
        draw.text((box[0], box[3]), text, fill="white", width=13, font=ImageFont.truetype("arial.ttf", 20))

    output_image_path = os.path.join(save_path, file)
    abnormimage.save(output_image_path)
    
    all_time = time.time() - start_time
    print(f'总用时{all_time}s')


          



  

