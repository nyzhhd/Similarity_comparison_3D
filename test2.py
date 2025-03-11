"""
word文档中对应的小部件相似度对比：

找到abnorm_path里面的图像文件  
去掉文件名中的数字得到对应的norm_path文件夹里面的标准图
读取xml,相同的box[4]送入sim.abnorm_detect_3D进行相似度比较
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

def extract_boxes_from_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    score=1

    boxes = []
    for object in root.findall('object'):
        class_name = object.find('name').text
        
        bndbox = object.find('bndbox')
        x_min = float(bndbox.find('xmin').text)
        y_min = float(bndbox.find('ymin').text)
        x_max = float(bndbox.find('xmax').text)
        y_max = float(bndbox.find('ymax').text)

        box = [x_min, y_min, x_max, y_max, class_name, score]
        boxes.append(box)
    # print(boxes)

    return boxes

if __name__ == "__main__":
    #img_path = r"E:\N-T2-20230531-CRH380AL2621-07-A-02-L-SSQZ-01-04.jpg"
    import xml.etree.ElementTree as ET
    import os
    from Sim_3D import Sim_3D
    sim=Sim_3D()

    # # +++++++++++++++++++++++++++++++ 通过检测获得box +++++++++++++++++++++++++++++
    #正常1 N-T1-20240109-CRH380AL2585-02-A-01-L-DJZC-04-08.jpg
    #异常1 N-T2-20230603-CRH380AL2614-02-B-02-L-DJZC-04-08.jpg
    #异常2 N-T1-20230603-CRH380AL2614-02-B-01-0-DJZC-09-11.jpg
    # abnormfile =    r"D:\adavance\Pyqt-main\DetectionModules\Algo1\Similarity_comparison_3D\test_data\N-T1-20240109-CRH380AL2585-02-A-01-L-DJZC-04-08.jpg"#第一张数据的位置，一般是异常数据的位置
    # normfile   =    r'D:\adavance\Pyqt-main\DetectionModules\Algo1\Similarity_comparison_3D\test_data\N-T1-20230603-CRH380AL2614-02-B-01-0-DJZC-09-11.jpg'#第二张数据的位置，一般是正常数据的位置
    # abnormimage = Image.open(abnormfile)
    # normimage = Image.open(normfile)
    
    # # 检测部件，若不分离异物检测，则也检测异物
    # test_boxs1, result_image1 = net_item_380.detect_results(abnormimage)
    # result_image1.show()
    # # print('111111111111111',test_boxs1)
    # test_boxs2, result_image2 = net_item_380.detect_results(normimage)
    # result_image2.show()
    # # print('22222222222222',test_boxs2)

    # confidence=0
    # for box in test_boxs1:  #取注油堵置信度最大的框
    #     if box[4]=='OilPlugS' or box[4]=='lost_OilPlugS' :
    #         if confidence<box[5]:
    #             abnormbox=box
    #             confidence=box[5]
    # confidence=0

    # for box in test_boxs2:  #取注油堵置信度最大的框
    #     if box[4]=='OilPlugS' or box[4]=='lost_OilPlugS' :
    #         if confidence<box[5]:
    #             normbox=box
    #             confidence=box[5]

    # sim.abnorm_detect_3D(abnormfile,abnormbox,normfile,normbox,0.8)

    # +++++++++++++++++++++++++++ 通过xml获得box ++++++++++++++++++++++++++++++++++
    norm_path=   r"D:\adavance\Similarity_comparison_3D\test_data\biaozhun" #标准图像所在的文件夹
    abnorm_path=   r"D:\adavance\Similarity_comparison_3D\test_data\daijian"#待检测图像所在的文件夹
    save_path=   r"D:\adavance\Similarity_comparison_3D\test_data\result"   #输出图像所在的文件夹

    import os
    import re
    import time
    ii=0
    start_time = time.time()
    for file in os.listdir(abnorm_path):#遍历待检测图像
        if file.endswith(".jpg"):
            abnorm_file = os.path.join(abnorm_path, file)      #待检测图像的路径 
            file_name = os.path.splitext(os.path.basename(abnorm_file))[0]
            letters_only = re.sub(r'\d', '', file_name)
            norm_file = norm_path + "\\" + letters_only + '.jpg'#推出标准图像路径 
            # print(abnorm_file)
            # print(norm_file)
            
            xml_abnormfile = os.path.splitext(abnorm_file)[0]+'.xml' 
            xml_normfile = os.path.splitext(norm_file)[0]+'.xml' 
            abnorm_boxes = extract_boxes_from_xml(xml_abnormfile)
            norm_boxes = extract_boxes_from_xml(xml_normfile)
            
            result_box=[]
            icp_error_list=[]
            for abnormbox in abnorm_boxes:  #遍历该待检测图像中所有的框
                abnormbox[0]=abnormbox[0]-10
                abnormbox[1]=abnormbox[1]-10
                abnormbox[2]=abnormbox[2]+10
                abnormbox[3]=abnormbox[3]+10

                for normbox_candidate in norm_boxes:#找到标准图像中相同的box[4]
                    if normbox_candidate[4] == abnormbox[4]:
                        normbox = normbox_candidate
                        normbox[0]=normbox[0]-10
                        normbox[1]=normbox[1]-10
                        normbox[2]=normbox[2]+10
                        normbox[3]=normbox[3]+10
                        break

                ii=ii+1
                print(f"-----------------------{ii}--------------------------")
                print(abnormbox)
                print(normbox)

                Sim_3D_dis_thresh=1.4   #距离阈值
                result_compare,icp_error=sim.abnorm_detect_3D(norm_file, normbox, abnorm_file, abnormbox, Sim_3D_dis_thresh)#调用类中方法，#点云降采样系数可采取默认，不必输入
                icp_error_list.append(icp_error)
                if result_compare==0:
                    result_box.append([abnormbox[0],abnormbox[1],abnormbox[2],abnormbox[3],'invalid data',abnormbox[5],icp_error])
                    print('该区域点云无效')
                elif result_compare==1:
                    print('点云平均距离相差较小，可认为大概率是正常的部件')
                    result_box.append([abnormbox[0],abnormbox[1],abnormbox[2],abnormbox[3],abnormbox[4],'norm',abnormbox[5],icp_error])
                elif result_compare==2:
                    print('点云平均距离相差较大，可认为大概率有缺陷的部件')
                    result_box.append([abnormbox[0],abnormbox[1],abnormbox[2],abnormbox[3],abnormbox[4],'abnorm',abnormbox[5],icp_error])
                print()
                
            #绘制结果图像保存到文件夹
            abnormimage = Image.open(abnorm_file)
            # Create a draw object
            draw = ImageDraw.Draw(abnormimage)
            text = f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            draw.text((30,30), text, fill="red", width=10,font=ImageFont.truetype("arial.ttf", 40))

            # Draw rectangles on the image based on result_box and display box4 and box5
            for box in result_box:
                draw.rectangle([box[0], box[1], box[2], box[3]], outline="red", width=2)
                # Display box4 and box5 alongside the rectangle
                text = f"type: {box[4]}\nresult: {box[5]}\ndistance:{box[7]}"
                draw.text((box[0], box[3]), text, fill="white", width=13, font=ImageFont.truetype("arial.ttf", 20))

            output_image_path = os.path.join(save_path, file)
            abnormimage.save(output_image_path)

           # Show the image
           #abnormimage.show()
    all_time = time.time() - start_time
    print(f'测试了{ii}个位点，平均距离{sum(icp_error_list)/len(icp_error_list)}，总用时{all_time}s,平均用时{all_time/ii}s')


          



  

