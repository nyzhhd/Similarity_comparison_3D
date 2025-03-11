import cv2
import numpy as np

from BoltHead_loose import *
from tqdm import tqdm
from collections import Counter
from sklearn.cluster import DBSCAN
from scipy.spatial import procrustes
from torchvision import transforms as T
#部件裁剪文件夹配置文件


#TODO:将配准获取的结果框与标准图上的框进行匹配，查看哪些框是带有_s的，返回_s的框即为螺栓松动测量的框

# 定义一个onnx目标检测器的类
class OnnxDetector(object):
    # 初始化
    def __init__(self, onnx_path, classes_path):
        self.sess = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])
        self.sess.set_providers(['CUDAExecutionProvider'], [{'device_id': 0}])
        with open(classes_path, 'r') as f:
            names = f.readlines()
        self.class_names = np.array([name.replace('\n', '').replace('\t', '') for name in names])
        self.format_transfer = T.Compose([T.ToTensor(), T.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.])])

    # 运行
    def run(self, img):
        # 获取原始图像的高度和宽度
        h_orig, w_orig = img.shape[:2]
        # 进行图像缩放
        scale = 1280
        size_divisor = 64
        # 计算缩放因子，保持宽高比并限制在指定的尺寸范围内
        scale_factor = min(scale / h_orig, scale / w_orig)
        # 缩放图像
        img_resize = cv2.resize(img, dsize=None, fx=scale_factor, fy=scale_factor)
        h_resize, w_resize = img_resize.shape[:2]
        # 计算需要补齐的像素数，使宽高都是size_divisor的整数倍
        right_pad = int(np.ceil(w_resize / size_divisor) * size_divisor - w_resize)
        bottom_pad = int(np.ceil(h_resize / size_divisor) * size_divisor - h_resize)
        # 使用常数值填充边界
        img_padding = cv2.copyMakeBorder(img_resize, 0, bottom_pad, 0, right_pad, cv2.BORDER_CONSTANT,
                                         value=(114, 114, 114))
        # 将填充后的图像转换为模型输入所需的格式
        img_tensor = self.format_transfer(img_padding).unsqueeze(0).numpy()
        # 运行模型获取预测结果
        num_dets, pred_boxes, pre_scores, pred_labels = self.sess.run(
            ["num_dets", "boxes", "scores", "labels"],
            {"images": img_tensor})
        # 处理模型输出结果
        num_dets = int(num_dets[0])
        pred_boxes = np.array(pred_boxes[0][:num_dets]) / scale_factor  # 缩放回原图大小
        pre_scores = np.array(pre_scores[0][:num_dets]).reshape(-1, 1)
        pred_labels = np.array(pred_labels[0][:num_dets])
        pred_label_names = self.class_names[pred_labels]
        # 构建目标列表
        target_list = []
        for i in range(pred_boxes.shape[0]):
            target = list(pred_boxes[i])
            target.append(pred_label_names[i])
            target.append(pre_scores[i][0])
            target_list.append(target)
        # 过滤置信度低于阈值的目标
        target_list = self.filter_confidence(target_list)

        return target_list

    def filter_confidence(self, target_list):
        """
  根据置信度阈值过滤目标列表。

  Parameters:
  - 包含目标信息的列表

  Returns:
  - 过滤后的目标列表
  """
        # 置信度阈值字典，针对每个类别设定阈值
        confidence_threshold = {
            'lost_BoltHead': 0.5,
            'Putty': 0.5,
            'lost_OilPlugS': 0.4,
            'BoltHead': 0.3,
            'OilPlugB': 0.5,
            'Mirror': 0.4,
            'OilLevelMirror': 0.4,
            'WholeCotterPin': 0.4,
            'MagneticBoltHolder': 0.4,
            'Clamp': 0.5
        }

        # 存储需要移除的目标的索引
        pop_indexs = []

        # 遍历目标列表
        for i in range(len(target_list)):
            # 如果目标的类别在置信度阈值字典中
            if target_list[i][4] in confidence_threshold:
                # 如果目标的置信度低于阈值
                if target_list[i][5] < confidence_threshold[target_list[i][4]]:
                    # 将目标索引添加到需要移除的列表中
                    pop_indexs.append(i)

        # 从目标列表中移除低置信度的目标
        for i in pop_indexs[::-1]:
            target_list.pop(i)

        return target_list

    # def draw_img(self, target_list, img):
    #     # 循环遍历目标列表并绘制矩形、类别和置信度
    #     for target in target_list:
    #         # 提取坐标和类别信息
    #         x_min, y_min, x_max, y_max, category, confidence = target[:6]
    #         # 将坐标转换为浮点数
    #         x_min, y_min, x_max, y_max, confidence = map(float, [x_min, y_min, x_max, y_max, confidence])
    #         # 在图像上绘制矩形
    #         cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
    #         # 构建显示文本
    #         label = f'{category}: {confidence:.2f}'
    #
    #         # 在图像上绘制类别和置信度文本
    #         cv2.putText(img, label, (int(x_min), int(y_min) + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 0), 2)
    #     # 显示图片
    #     cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    #     cv2.imshow('img', img)
    #     # cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    #
    #     return 0


# 定义LoFTR关键点检测类
class loftrInfer(object):
    ''' 初始化之后，仅需调用run函数 '''

    def __init__(self, model_path="weights/indoor_ds.ckpt", match_thr=0.2):
        '''
      初始化，输入参数:
              model_path: 模型地址
  '''
        default_cfg['match_coarse']['thr'] = match_thr  # 修改关键点匹配的阈值
        self.matcher = LoFTR(config=default_cfg)  # 初始化模型
        self.matcher.load_state_dict(torch.load(model_path)['state_dict'])  # 下载训练好的模型文件，可选indoor_ds 、outdoor_ds
        self.matcher = self.matcher.eval().cuda()  # cuda验证

    def _infer_run(self, img0_raw, img1_raw):
        '''
      推理单对图片，输入参数:
              img0_raw 、img1_raw    numpy.ndarray类型，单通道图像
              返回值:
              np_result/False     False 或 (n,5)推理结果，numpy.ndarray类型； 格式为(p1x,p1y,p2x,p2y,conf)

  '''
        img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.  # 转torch格式，cuda ，归一到0-1
        img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.
        batch = {'image0': img0, 'image1': img1}  # 模型输入为字典，加载输入

        # Inference with LoFTR and get prediction 开始推理
        with torch.no_grad():
            self.matcher(batch)  # 网络推理
            mkpts0 = batch['mkpts0_f'].cpu().numpy()  # (n,2) 0的结果 -特征点
            mkpts1 = batch['mkpts1_f'].cpu().numpy()  # (n,2) 1的结果 -特征点
            mconf = batch['mconf'].cpu().numpy()  # (n,)      置信度

        # 筛选，需要四个以上的匹配点才能得到单应性矩阵
        if mconf.shape[0] < 4:
            return False
        mconf = mconf[:, np.newaxis]  # 末尾增加新维度
        np_result = np.hstack((mkpts0, mkpts1, mconf))  # 水平拼接
        # print(np_result.shape)
        list_result = list(np_result)

        def key_(a):
            return a[-1]

        list_result.sort(key=key_, reverse=True)  # 按得分从大到小排序
        np_result = np.array(list_result)
        return np_result

    def _points_filter(self, np_result, lenth=200, use_DBSCAN=True):
        '''
      进行特征值筛选，输入参数:
              np_result  推理结果(n,5)
              lenth       -1   不进行筛选，取全部
                          >0  取前nums个
              use_kmeans   bool类型: 0 - 不使用
                                  1 -  使用聚类，取最多一类
  '''
        lenth = min(lenth, np_result.shape[0])  # 选最大200个置信度较大的点对
        if lenth < 4: lenth = 4

        mkpts0 = np_result[:lenth, :2].copy()
        mkpts1 = np_result[:lenth, 2:4].copy()

        if use_DBSCAN:
            # 删除不合理的匹配点
            angle = np.tan((mkpts0[:, 1] - mkpts1[:, 1]) / (500 + mkpts0[:, 0] - mkpts1[:, 0]))
            db = DBSCAN(eps=0.1, min_samples=50)
            db.fit(angle.reshape(-1, 1))
            y_labels = db.labels_
            # y_labels_unique = np.unique(y_labels)
            use_mkpts0 = mkpts0[y_labels != -1]
            use_mkpts1 = mkpts1[y_labels != -1]

            if use_mkpts0.shape[0] < 4:
                # print(f'使用DBSCAN过滤后的关键点太少！')
                return mkpts0, mkpts1
            return use_mkpts0, use_mkpts1
        return mkpts0, mkpts1

    def run(self, img0_bgr, img1_bgr, lenth=200, use_DBSCAN=True):
        '''
      只需要调用该函数，完成推理+拼接
      输入参数 :
          img0_bgr , img1_bgr  彩色图像，默认左右
          lenth       -1   不进行筛选，取全部
                      >0  取前nums个
          use_DBSCAN   bool类型: 0 - 不使用
                              1 -  使用聚类，取最多一类
      返回值:
          mkpts0 img0上的匹配点
          mkpts1 img1上的匹配点
  '''
        img0_raw = cv2.cvtColor(img0_bgr, cv2.COLOR_BGR2GRAY)  # 转灰度，网络输入的是单通道图
        img1_raw = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY)

        np_result = self._infer_run(img0_raw, img1_raw)  # 推理
        # print(np_result.shape)
        if np_result is False:
            print("特征点数量不够！！！")
            return False
        mkpts0, mkpts1 = self._points_filter(np_result, lenth=lenth, use_DBSCAN=use_DBSCAN)  # 特征点筛选

        return mkpts0, mkpts1

    # 绘制
    def draw_img(self, img0_bgr, img1_bgr, mkpts0, mkpts1, color=(0, 200, 0), text='.'):
        combined_image = np.hstack((img0_bgr, img1_bgr))
        for point0, point1 in zip(mkpts0, mkpts1):
            cv2.circle(combined_image, (int(point0[0]), int(point0[1])), 1, color, -1)
            cv2.circle(combined_image, (int(point1[0]) + img0_bgr.shape[1], int(point1[1])), 1, (255, 193, 37), -1)
            # 在图像上绘制文本
            cv2.putText(combined_image, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            # 绘制直线
            # cv2.line(combined_image, (int(point0[0]), int(point0[1])), (int(point1[0]) + img0_bgr.shape[1], int(point1[1])), (100, 100, 0),
            #          1)
        # 显示图片
        cv2.namedWindow('Combined Image with Keypoints', cv2.WINDOW_NORMAL)
        cv2.imshow('Combined Image with Keypoints', combined_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return combined_image


# 通过斜率来筛选点组
def slope_screening(mkpts0, mkpts1, slope_difference_threshold=0.01):
    # 计算直线的斜率
    slope_list_w = []
    slope_list_h = []
    for point0, point1 in zip(mkpts0, mkpts1):
        slope_w = abs(point0[1] - point1[1]) / abs(1000 + point0[0] - point1[0])
        slope_list_w.append(round(slope_w, 2))
        slope_h = abs(point0[0] - point1[0]) / abs(1000 + point0[1] - point1[1])
        slope_list_h.append(round(slope_h, 2))
    # print(f'slope_list_w:{slope_list_w}')
    # print(f'slope_list_h:{slope_list_h}')
    mode_w = [k for k, v in Counter(slope_list_w).items() if v == max(Counter(slope_list_w).values())][0]
    mode_h = [k for k, v in Counter(slope_list_h).items() if v == max(Counter(slope_list_h).values())][0]

    # print(f'mode_w:{mode_w}')
    # print(f'mode_h:{mode_h}')
    fault_points_list = []
    slope_list_w_list = []
    slope_list_h_list = []
    for point0, point1 in zip(mkpts0, mkpts1):
        # 删除斜率与全部斜率中位数相差过大的点
        if len(slope_list_w) > 0:
            slope_difference_w = abs(abs((point0[1] - point1[1]) / (1000 + point0[0] - point1[0])) - abs(mode_w))
            slope_difference_w = round(slope_difference_w, 2)
            slope_list_w_list.append(slope_difference_w)

            slope_difference_h = abs(
                abs(point0[0] - point1[0]) / abs((1000 + point0[1] - point1[1])) - abs(mode_h))
            slope_difference_h = round(slope_difference_h, 2)
            slope_list_h_list.append(slope_difference_h)

            if slope_difference_w > slope_difference_threshold or slope_difference_h > slope_difference_threshold:
                fault_points_list.append((point0[0], point0[1]))
                fault_points_list.append((point1[0], point1[1]))
        else:
            print(f'关键点数量不够')
            # time.sleep(100)

    # print(f'斜率差值w:{slope_list_w_list}')
    # print(f'斜率差值h:{slope_list_h_list}')
    # 定义fault_points
    fault_points = fault_points_list

    # 删除fault_points中出现的点
    clean_mkpts0 = []
    clean_mkpts1 = []
    for point0, point1 in zip(mkpts0, mkpts1):
        if tuple(point0) not in fault_points and tuple(point1) not in fault_points:
            clean_mkpts0.append(point0)
            clean_mkpts1.append(point1)
    clean_mkpts0 = np.array(clean_mkpts0)
    clean_mkpts1 = np.array(clean_mkpts1)

    return clean_mkpts0, clean_mkpts1


# 读取xml
def read_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    data_list = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        item_data = [xmin, ymin, xmax, ymax, name]

        if "BoltHead" in item_data[-1]:
            data_list.append(item_data)
            print("item_data++++++++++++", item_data)
        else:
            continue

    return data_list


def draw_yolo_bbox(target_list, image, x_show_offset=0, color=(0, 255, 0), is_save=False):
    # 循环遍历目标列表并绘制矩形、类别和置信度
    for target in target_list:
        # 将前四个元素转换为浮点数格式
        x_min = float(target[0])
        y_min = float(target[1])
        x_max = float(target[2])
        y_max = float(target[3])
        x_min += x_show_offset
        x_max += x_show_offset

        try:
            category = target[4]
        except:
            category = 'None'

        # 提取坐标和类别信息
        try:
            confidence = float(target[5])
        except:
            confidence = 0

        # 将坐标转换为浮点数
        x_min, y_min, x_max, y_max, confidence = map(float, [x_min, y_min, x_max, y_max, confidence])

        # 在图像上绘制矩形
        cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)

        # 构建显示文本
        label = f'{category}: {confidence:.2f}'

        # 在图像上绘制类别和置信度文本
        cv2.putText(image, label, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 显示绘制结果
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if is_save:
        index = 0
        filename = os.path.join(f"result/{index}.jpg")
        while os.path.exists(filename):
            index += 1
            filename = os.path.join(f"result/{index}.jpg")
        cv2.imwrite(filename, image)

    return image


# 坐标缩放
def scale_bboxes(target_list, scale_para):
    scaled_list = []
    for item in target_list:
        try:
            scaled_item = [
                int(item[0]) * scale_para,  # 缩放xmin
                int(item[1]) * scale_para,  # 缩放ymin
                int(item[2]) * scale_para,  # 缩放xmax
                int(item[3]) * scale_para,  # 缩放ymax
                item[4],  # 其他元素保持不变
                item[5]
            ]
        except:
            scaled_item = [
                int(item[0]) * scale_para,  # 缩放xmin
                int(item[1]) * scale_para,  # 缩放ymin
                int(item[2]) * scale_para,  # 缩放xmax
                int(item[3]) * scale_para,  # 缩放ymax
                item[4],  # 其他元素保持不变
            ]
        scaled_list.append(scaled_item)
    return scaled_list


# 找到bbox内的点的索引
def research_points_index(mkpts, bbox_position, padding=20):
    x_values = mkpts[:, 0]
    y_values = mkpts[:, 1]

    selected_indices = np.logical_and.reduce((x_values >= bbox_position[0] - 10,
                                              y_values >= bbox_position[1] - 10,
                                              x_values <= bbox_position[2] + 10,
                                              y_values <= bbox_position[3] + 10))
    return selected_indices


# 找到同一类别中覆盖目标点数最多的那一个框并返回
def research_target_bbox(target_category, bbox_mkpts, std_xml_sacled, detect_list, mkpts_std, mkpts_test):
    fault_test_bbox = []
    correct_std_bbox = []
    correct_test_bbox = []

    target_category = target_category
    # print(f'target_category:{target_category}')
    # 筛选出类别为目标类别的坐标框
    detect_list = np.array(detect_list)
    # 1、没有检测框没有该类别（该类别全部丢失）
    if any([item[4] == target_category for item in detect_list]) != True:
        # print(f'{target_category}丢失！')
        # 将标准图中该类别的框移除
        std_xml_sacled_list = [item for item in [std_xml_sacled] if target_category not in item[4]]
        # print(f'std_xml_sacled_list:{std_xml_sacled_list}')

        # 进行配准
        result = [0, 0, 0, 0, target_category + '_LOST', 88.88]
        return result

    filtered_boxes = detect_list[detect_list[:, 4] == target_category][:, :4]
    # 将坐标框转换为浮点数类型
    filtered_boxes = filtered_boxes.astype(float)

    # 统计每个坐标框覆盖的点数
    covered_points_count = []
    for box in filtered_boxes:
        x1, y1, x2, y2 = box
        count = ((x1 <= bbox_mkpts[:, 0]) &
                 (bbox_mkpts[:, 0] <= x2) &
                 (y1 <= bbox_mkpts[:, 1]) &
                 (bbox_mkpts[:, 1] <= y2)).sum()
        covered_points_count.append(count)
    print("covered_points_count:",covered_points_count)

    # 如果没有点集覆盖
    if sum(covered_points_count) < 2:
        print(f'没有LoFTR的关键点！')
        print(f'std_xml_sacled_list:{std_xml_sacled}')
        result = [0, 0, 0, 0, target_category + '_ERROR', 0]
        return result

    # 找到覆盖点数最多的坐标框的索引
    # print(f'covered_points_count:{covered_points_count}')
    max_covered_index = np.argmax(covered_points_count)
    result = filtered_boxes[max_covered_index].tolist()
    result.append(target_category)
    result.append(max(covered_points_count) if len(covered_points_count) > 0 else 0)

    return result


# 删除重复匹配的框
def remove_duplicate_matching(result_list):
    removed_indexes = []  # 记录被删除的索引

    for i in range(len(result_list)):
        for j in range(i + 1, len(result_list)):
            if result_list[i][:5] == result_list[j][:5] and 'Error' not in result_list[i][4] and 'Lost' not in \
                    result_list[i][4]:  # 如果前5个元素相同
                if result_list[i][-1] < result_list[j][-1]:  # 删除最后一个元素较小的项
                    removed_indexes.append(i)
                    break
                else:
                    removed_indexes.append(j)
    removed_indexes = list(set(removed_indexes))  # 去除重复索引
    for index in sorted(removed_indexes, reverse=True):  # 从后往前删，避免索引错位
        del result_list[index]

    return result_list, removed_indexes


# 坐标框映射
def removed_bbox_to_test_bbox(new_std_bbox, result_list, removed_std_bbox, mkpts_std, mkpts_test):
    new_std_bbox = np.array(new_std_bbox)[:, :4].astype(float)
    result_list_bbox = np.array(result_list)[:, :4].astype(float)
    # print(f'new_std_bbox:{new_std_bbox}')
    std_points_num = len(np.reshape(new_std_bbox[:, :4].astype(float), (-1, 2)))
    # print(f'std_points_num:{std_points_num}')
    # print(
    #     f'np.reshape(new_std_bbox[:, :4].astype(float), (-1, 2)):{np.reshape(new_std_bbox[:, :4].astype(float), (-1, 2))}')
    test_points_num = len(np.reshape(result_list_bbox[:, :4].astype(float), (-1, 2)))
    # print(f'test_points_num:{test_points_num}')
    # print(
    #     f'np.reshape(result_list_bbox[:, :4].astype(float), (-1, 2)):{np.reshape(result_list_bbox[:, :4].astype(float), (-1, 2))}')

    if std_points_num >= 2:
        # 计算标准边界框到处理后的二维list的变换矩阵
        # print(f'使用矩形框的映射矩阵')

        # 遍历五个元素的所有组合
        combinations = list(itertools.combinations(new_std_bbox, len(result_list_bbox)))

        all_list = []
        # 输出每种组合
        for i, c in enumerate(combinations):
            c = list(c)

            result_arr = np.array(result_list_bbox)
            std_arr = np.array(c)

            # print(f'std_arr:{std_arr}')
            # print(f'result_arr:{result_arr}')
            # print('----------------------')

            # 使用Procrustes分析计算相似度
            if len(result_arr) > 1:
                _, _, disparity = procrustes(result_arr, std_arr)
            else:
                disparity = 1
            # 计算相似度（范围为0到1，越接近1表示越相似）
            similarity = 1 - disparity

            # print(f"del {i} 相似度：{similarity}")

            # 输出每种组合在arr中的索引
            combination_indices = []
            # print(f'new_std_bbox：{new_std_bbox}')
            # print(f'c：{np.array(c)}')
            for i, sublist in enumerate(new_std_bbox):
                if sublist in np.array(c):
                    combination_indices.append(i)
            all_list.append([std_arr, result_arr, similarity, i, combination_indices])
        # 使用lambda函数定义排序规则，按每一项最后一个元素的大小进行排序
        sorted_array = sorted(all_list, key=lambda x: x[2])
        # print(f'sorted_array:{sorted_array}')
        # print('----------------------------------------------------')
        # print(f'sorted_array[0]:{sorted_array[0]}')

        M, _ = cv2.findHomography(np.reshape(new_std_bbox[:, :4].astype(float), (-1, 2)),
                                  np.reshape(result_list_bbox[:, :4].astype(float), (-1, 2)))
    else:
        # print(f'使用LoFTR映射矩阵！')
        M, _ = cv2.findHomography(mkpts_std, mkpts_test)
        # print(f'LoFTR映射矩阵M:{M}')
    # 定义需要进行映射的点
    removed_std_bbox_tmp = np.array(removed_std_bbox)[:, :4].astype(float)
    # print(f'removed_std_bbox_tmp:{removed_std_bbox_tmp}')
    # 对移除的标准边界框进行坐标映射
    mapped_points = cv2.perspectiveTransform(np.array([np.reshape(removed_std_bbox_tmp, (-1, 2))[:, :2]]), M)
    mapped_points = np.reshape(mapped_points, (-1, 1, 4))[0]
    # print(f'mapped_points:{mapped_points}')
    # 替换元素
    tmp_bbox = np.array(removed_std_bbox)
    # print(f'removed_std_bbox:{removed_std_bbox}')
    tmp_bbox[:, :4] = mapped_points
    removed_test_bbox = tmp_bbox
    # print(f'removed_test_bbox:{removed_test_bbox}')
    # 在每个项中添加新元素0
    # 遍历每个项，并在每个项中添加一个元素0
    result_array = []
    for item in removed_test_bbox:
        item = np.append(item, 0)
        result_array.append(item)
    removed_test_bbox = np.array(result_array)
    # print(f'removed_test_bbox:{removed_test_bbox}')

    # 将新数组添加到原始数组中
    result_list = np.concatenate((result_list, removed_test_bbox), axis=0)

    return result_list


# 对于目标检测框的IOU过滤
def filter_boxes(yolo_detect_list):
    # 定义一个字典，用于存储每个类别的物体框
    boxes_dict = {}
    for box in yolo_detect_list:
        if box[4] not in boxes_dict:
            boxes_dict[box[4]] = [box]
        else:
            boxes_dict[box[4]].append(box)

    filtered_boxes = []

    # 遍历每个类别的物体框
    for boxes in boxes_dict.values():
        # 计算每个物体框的面积
        areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]

        # 遍历每个物体框，判断是否与其他物体框有重合或包围关系
        for i, box1 in enumerate(boxes):
            skip_box = False
            for j, box2 in enumerate(boxes):
                if i == j:
                    continue
                # 判断是否有重合或包围关系
                if (box1[0] <= box2[0] and box1[1] <= box2[1] and box1[2] >= box2[2] and box1[3] >= box2[3]) or \
                        (box1[0] >= box2[0] and box1[1] >= box2[1] and box1[2] <= box2[2] and box1[3] <= box2[3]):
                    # 如果面积较小的物体框已经被加入到filtered_boxes中，就跳过当前循环
                    if i != areas.index(min(areas[i], areas[j])):
                        skip_box = True
                        break

            if not skip_box:
                filtered_boxes.append(box1)

    return filtered_boxes


def calculate_iou(box1, box2):
    """
 计算两个框的交并比(IOU)
 :param box1: 第一个框的坐标[x_min, y_min, x_max, y_max]
 :param box2: 第二个框的坐标[x_min, y_min, x_max, y_max]
 :return: 两个框的IOU值
 """
    # 计算交集的坐标
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    # 计算交集的面积
    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)

    # 计算两个框各自的面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 计算并集的面积
    union_area = box1_area + box2_area - intersection_area

    # 计算交并比
    iou = intersection_area / union_area

    return iou


def find_corresponding_boxes(result_list, yolo_detect_list):
    # 使用 tolist() 方法将其转换为 Python 列表
    result_list = result_list.tolist()
    # yolo_detect_list = yolo_detect_list.tolist()
    # 初始化结果列表
    result_with_labels = []

    # 遍历result_list中的每个框
    for result_box_all in result_list:
        result_box = list(map(float, result_box_all[:4]))
        # 初始化最大IOU值和对应的类别
        max_iou = 0
        corresponding_label = None
        tmp_result_box = []
        # 遍历std_xml_scaled_list中的每个框
        for detect_box in yolo_detect_list:
            # 计算IOU值
            if result_box_all[4] == detect_box[4]:
                iou = calculate_iou(result_box, detect_box[:4])
            else:
                iou = 0
            # 如果当前IOU值大于之前的最大IOU值，则更新最大IOU值和对应的类别
            if iou > max_iou:
                max_iou = iou
                if iou > 0.15:
                    corresponding_label = detect_box[4]
                    confidence_degree = detect_box[5]
                    tmp_result_box = detect_box[:4]

        # print(f'max_iou:{max_iou}')
        if corresponding_label == None:
            tmp_result_box = result_box[:4]
            corresponding_label = result_box_all[4] + '_Lost'
            confidence_degree = 0
        # 将结果添加到结果列表，包括坐标和对应的类别
        result_with_labels.append(tmp_result_box + [corresponding_label] + [confidence_degree])

    return result_with_labels

def bolt_loose(filepath, test_x1, test_y1, test_x2, test_y2, depthmap):

    coordinates_list = []
    for i in range(test_x1, test_x2):
        for j in range(test_y1, test_y2):
            # print("i,j",i,j)
            bolt_x, bolt_y, bolt_z = accept_xyz(depthmap, i, j, p=0.1)
            coordinates_list.append([int(bolt_x), int(bolt_y), int(bolt_z)])
    # print("sorted_points_new------", coordinates_list)
    coordinates_list = [po for po in coordinates_list if po[2] != 0]

    sorted_points = coordinates_list

    # print("sorted_points_old------", len(sorted_points))
    # sorted_points = [point_0 for point_0 in coordinates_list if point_0[2] != 0]
    # print("sorted_points_new------",sorted_points)
    # 计算1%和99%的索引
    percentile_1 = int(0.1* len(sorted_points))
    percentile_99 = int(0.99 * len(sorted_points))
    # 选择5%到95%的点
    point_cloud = sorted_points[percentile_1:percentile_99]
    # 将数组列表转换为numpy列表
    point_cloud = np.array(point_cloud)
    print("point_cloud+++++++++++++++++++",point_cloud)
    point_cloud = random_downsample(point_cloud, 0.5)


    return  point_cloud


def bolt_loose_filter(filepath, test_x1, test_y1, test_x2, test_y2, depthmap):

    coordinates_list = []
    for i in range(test_x1, test_x2):
        for j in range(test_y1, test_y2):
            # print("i,j",i,j)
            bolt_x, bolt_y, bolt_z = accept_xyz(depthmap, i, j, p=0.1)
            coordinates_list.append([int(bolt_x), int(bolt_y), int(bolt_z)])
    # print("sorted_points_new------", coordinates_list)
    coordinates_list = [po for po in coordinates_list if po[2] != 0]

    sorted_points = coordinates_list

    # print("sorted_points_old------", len(sorted_points))
    # sorted_points = [point_0 for point_0 in coordinates_list if point_0[2] != 0]
    # print("sorted_points_new------",sorted_points)
    # 计算1%和99%的索引
    percentile_1 = int(0.1* len(sorted_points))
    percentile_99 = int(0.9 * len(sorted_points))
    # 选择5%到95%的点
    point_cloud = sorted_points[percentile_1:percentile_99]
    # 将数组列表转换为numpy列表
    point_cloud = np.array(point_cloud)
    point_cloud = random_downsample(point_cloud, 0.25)


    print(f"数量：{len(point_cloud)}")
    use_point = int(len(point_cloud) * 0.1)
    if use_point > 200:
        use_point = 300
    if use_point <= 200:
        use_point = 200


    # print("selected_points:",len(selected_points))
    # 拟合第一个平面
    bolt_dist, first_data, second_data = [], [], []
    io = 0
    second_plane_points_ind = []
    while True:
        io += 1
        print("io",io)
        first_plane_points_ind, first_data, first_plane_params = Ransac3d(point_cloud, use_point,1)
        print("第一个平面first_plane_points_ind-------------",len(first_plane_points_ind))
        # 找到所有不属于第一平面的点
        mask = np.ones(point_cloud.shape[0], bool)
        mask[first_plane_points_ind] = False
        mask[second_plane_points_ind] = False
        not_in_plane_points = point_cloud[mask]
        # 非一点的在原点云的索引
        temp_index = np.where(mask)[0]
        second_plane_points_ind, second_data, second_plane_params = Ransac3d(not_in_plane_points, use_point + 50,0.5)
        print("第2个平面first_plane_points_ind-------------", len(second_plane_points_ind))
        # 根据两个平面的参数计算平面夹角的余弦值
        x0, y0, z0, _ = first_plane_params
        sqrt_xyz = (x0 * x0 + y0 * y0 + z0 * z0) ** 0.5
        a, b, c, _ = second_plane_params
        sqrt_abc = (a * a + b * b + c * c) ** 0.5
        cos = np.fabs(a * x0 + b * y0 + c * z0) / (sqrt_abc * sqrt_xyz)
        print("jiaodu:",cos)
        # 计算两个平面之间的距离
        dists = get_dist_to_plane(first_plane_params,
                                  point_cloud[temp_index[second_plane_points_ind]])
        dists = [item for item in dists if 0.5 <item <15]
        sorted_dists = sorted(dists)
        lower_index = int(0.2*(len(sorted_dists)))
        upper_index = int(0.8*(len(sorted_dists)-1))
        select_value = sorted_dists[lower_index:upper_index]
        bolt_dist = np.mean(select_value)
        bolt_dist = round(bolt_dist,1)
        if io == 5:
            break
        if bolt_dist > 3 and cos > 0.95:
            break
    # bolt_dist = np.mean(dists[int(0.3 * dists.shape[0]):int(0.7 * dists.shape[0])])
        print(f"高度为{select_value}")
    return bolt_dist, first_data, second_data





def BoltHead_3DSIM_and_measure(test_img_path,std_img_path,test_box,std_box,sim_loose_thresh=1, dis_loose_thresh=0.6):
    #result_back=0 没有经过计算，或者计算错误
    #result_back=1 判断为正常部件
    #result_back=1 判断为松动、异常
    result_back=0

    test_img_xyz = test_img_path.replace("jpg", "xyz")
    std_img_xyz = std_img_path.replace("jpg", "xyz")
    depthmap_test = fun_readXYZ(test_img_xyz)
    depthmap_std = fun_readXYZ(std_img_xyz)

    coordinates_list = []
    coordinates_list_std = []
    try:
        test_x1, test_y1, test_x2, test_y2 = max(test_box[0] , 0), max(
            test_box[1]  , 0), min(test_box[2] , 1920), min(
            test_box[3] , 1200)
        std_x1, std_y1, std_x2, std_y2 = max(std_box[0] , 0), max(
            std_box[1]  , 0), min(std_box[2] , 1920), min(
            std_box[3] , 1200)

        
        first_point_cloud = bolt_loose(test_img_xyz, test_x1 - 10, test_y1 - 10 , test_x2 + 10 , test_y2 + 10, depthmap_test)
        # std_loose,first_datas, second_datas = bolt_loose(std_img_xyz, std_x1 - 5, std_y1 - 5, std_x2 + 5, std_y2 + 5, depthmap_std)
        second_point_cloud = bolt_loose(std_img_xyz, std_x1 -20, std_y1 - 20 , std_x2 + 20, std_y2 + 20, depthmap_std)
    except:
        print(f"{test_img_xyz}点云数量：{len(depthmap_test)}不足，跳过处理！！！")
        return result_back,-1,-1
    if len(first_point_cloud) > 100 and len(second_point_cloud) > 100:
        point_cloud1 = o3d.geometry.PointCloud()
        point_cloud1.points = o3d.utility.Vector3dVector(
            first_point_cloud)  # Replace inliersResult_data1 with your actual data
        point_cloud2 = o3d.geometry.PointCloud()
        point_cloud2.points = o3d.utility.Vector3dVector(
            second_point_cloud)  # Replace inliersResult_data2 with your actual data
        # point_cloud1, _ = point_cloud1.remove_radius_outlier(nb_points=1500, radius=100)
        # point_cloud2, _ = point_cloud2.remove_radius_outlier(nb_points=2000, radius=100)
        point_cloud1, _ = point_cloud1.remove_radius_outlier(nb_points=600, radius=10)
        point_cloud2, _ = point_cloud2.remove_radius_outlier(nb_points=600, radius=10)
        color_cloud1 = [[1, 0, 0] for i in range(len(point_cloud1.points))]  # 点云a的颜色为红色
        point_cloud1.colors = o3d.utility.Vector3dVector(color_cloud1)
        color_cloud2 = [[0, 0, 1] for i in range(len(point_cloud2.points))]  # 点云a的颜色为红色
        point_cloud2.colors = o3d.utility.Vector3dVector(color_cloud2)
        # o3d.visualization.draw_geometries([point_cloud1, point_cloud2], window_name='Initial Alignment')

        point1_center = np.asarray(point_cloud1.get_center())
        point2_center = np.asarray(point_cloud2.get_center())
        tr = point1_center - point2_center
        point_cloud2 = point_cloud2.translate(tr)

        icp_result = o3d.pipelines.registration.registration_icp(
            point_cloud1, point_cloud2, max_correspondence_distance=20,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint())
        point_cloud1 = point_cloud1.transform(icp_result.transformation)

        icp_result = o3d.pipelines.registration.registration_icp(
            point_cloud1, point_cloud2, max_correspondence_distance=1,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint())
        print("icp_result2", icp_result.fitness)
        # o3d.visualization.draw_geometries([point_cloud1, point_cloud2],window_name='Final Alignment')
        sim_loose_value = icp_result.fitness
    else:
        sim_loose_value = 0.3

    # if 0.5 <abs(loose_value)< 1:
    try:
        if abs(sim_loose_value)< sim_loose_thresh:
            std_loose, first_datas, second_datas = bolt_loose_filter(std_img_xyz, std_x1 - 2 ,
                                                                        std_y1,
                                                                        std_x2 + 2 , std_y2,
                                                                        depthmap_std)
            test_loose, first_data, second_data = bolt_loose_filter(test_img_xyz, test_x1-5, test_y1,
                                                                test_x2 + 5, test_y2, depthmap_test)
            test_loose = round(test_loose, 2)
            std_loose = round(std_loose, 2)

            print("test_loose:", test_loose)
            print("std_loose:", std_loose)
            dis_loose_value = test_loose - std_loose

            # if abs(loose_value) >= 0.5 and test_loose> 2 and std_loose> 2:
            if dis_loose_value >= dis_loose_thresh:
                result_back = 2 
                return result_back,sim_loose_value,dis_loose_value
            else:
                result_back = 1
                return result_back,sim_loose_value,dis_loose_value
        else:
            result_back=1
            return result_back,sim_loose_value,-1
    except:
        print("点云计算错误---------------------")
        return result_back,sim_loose_value,-1

if __name__ == "__main__":
    pass
