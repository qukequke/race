import os
import torch as t
from utils.config import opt
from model import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from data.util import  read_image
# from utils.vis_tool import vis_bbox
# from utils import array_tool as at
import glob
import cv2
import json
from utils.config import opt

def get_dict():
    fn = os.path.join(os.getcwd(), 'train_no_poly.json')
    f = open(fn, 'rb')
    file_dic = json.load(f)
    categories = file_dic['categories']
    id_to_cate = {}
    for i in categories:
        id_ = i['id']
        name = i['name']
        id_to_cate[id_] = name
    images = file_dic['images']
    fn_to_id = {}
    for i in images:
        fn = i['file_name']
        id_ = i['id']
        fn_to_id[fn] = id_

    annotation = file_dic['annotations']
    id_to_bbox = {}
    for i in annotation:
        id_ = i['image_id']
        bbox = i['bbox']
        bbox.append(i['category_id'])
        if id_ in id_to_bbox:
            id_to_bbox[id_].append(bbox)
        else:
            id_to_bbox[id_] = [bbox]

    fn_to_annotation = {}
    for i in fn_to_id.keys():
        im_id = fn_to_id[i]
        if id_to_bbox.get(im_id) == None:
            # print(im_id, '缺失')
            pass
        else:
            fn_to_annotation[i] = id_to_bbox.get(im_id)
    return fn_to_annotation
    # self.fns = list(self.fn_to_annotation.keys())
    # print('len=',len(self.fns))
    # self.dir = dir_
    # self.label_names = BBOX_LABEL_NAMES


faster_rcnn = FasterRCNNVGG16()
trainer = FasterRCNNTrainer(faster_rcnn).cuda()

# imgs = os.listdir('../VOCdevkit/VOC2007/JPEGImages')
# img = imgs[5]
# img = read_image('../VOCdevkit/VOC2007/JPEGImages/' + img)
# img = t.from_numpy(img)[None]

VOC_BBOX_LABEL_NAMES = (
    'Iron',
    'black',
    'knife',
    'power',
    'scissors',
)

if not os.path.exists("../submit/"):
    os.mkdir("../submit/")

if not os.path.exists("../outputs/"):
    os.mkdir("../outputs/")

# trainer.load('/home/sar/AnnotatedNetworkModelGit-master/fasterrcnn/simple-faster-rcnn/fasterrcnn_03241017')
trainer.load(opt.load_path)

opt.caffe_pretrain=False# this model was trained from caffe-pretrained model

filenames = glob.glob('../jinnan2_round1_train_20190305/restricted/*.jpg')

results_file = open("../submit/larger%s.csv"%str(0.5),"w")
iii = 1
save_path = '../test_train'
# import matplotlib.pyplot as plt
for filename in filenames[:5]:
    img = read_image(filename)
    img = t.from_numpy(img)[None]
    fn_to_anno = get_dict()
    basename = os.path.basename(filename)
    anno = fn_to_anno.get(os.path.basename(filename))
    _bboxes, _labels, _scores = trainer.faster_rcnn.predict(img,visualize=True)
    # print(_bboxes)
    bboxes = _bboxes[0]
    labels = _labels[0]
    scores = _scores[0]

    image = cv2.imread(filename)
    image2 = cv2.imread(filename)

    if anno== None:
        cv2.imwrite(os.path.join(save_path, filename), image2)
    else:
        for bbox_and_cat in anno:
            xmin, ymin = int(bbox_and_cat[0]), int(bbox_and_cat[1])
            xmax, ymax = int(bbox_and_cat[0] + bbox_and_cat[2]), int(bbox_and_cat[1] + bbox_and_cat[3])
            category = bbox_and_cat[4]
            # print(category)
            cv2.rectangle(image2, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
            cv2.putText(image2, VOC_BBOX_LABEL_NAMES[category-1], (xmin, ymin), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imwrite(os.path.join(save_path, basename[:-4]+'_label'+basename[-4:]), image2)
        # i += 1

    # fig, ax = plt.subplots(1,2,figsize=(5,10))
    # for i,box in (enumerate(anno)):
    #     cv2.rectangle(image,(int(box[1]),int(box[0])),(int(box[3]),int(box[2])),color=(0,0,255),thickness=2 )
    #     cv2.putText(image,VOC_BBOX_LABEL_NAMES[labels[i]],(int(box[1]), int(box[0])),fontFace=cv2.FONT_HERSHEY_COMPLEX,fontScale=0.5,color=(0,0,255), thickness=1)
    #     # cv2.imshow('aa', image)
    #     # cv2.waitKey(0)
    #     results_file.write(filename.split("/")[-1] +","+ str(int(box[1])) + " " + str(int(box[0])) +  " " + str(int(box[3])) +' ' +str(int(box[2]))+','+str(labels[i]) +','+str(scores[i])+ "\n")

    for i,box in (enumerate(bboxes)):
        if i == 0:
            for bbox_and_cat in anno:
                xmin, ymin = int(bbox_and_cat[0]), int(bbox_and_cat[1])
                xmax, ymax = int(bbox_and_cat[0] + bbox_and_cat[2]), int(bbox_and_cat[1] + bbox_and_cat[3])
                category = bbox_and_cat[4]
                # print(category)
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        cv2.rectangle(image,(int(box[1]),int(box[0])),(int(box[3]),int(box[2])),color=(0,0,255),thickness=2 )
        cv2.putText(image,VOC_BBOX_LABEL_NAMES[labels[i]]+':'+str(scores[i]),(int(box[1]), int(box[0])),fontFace=cv2.FONT_HERSHEY_COMPLEX,fontScale=0.5,color=(0,0,255), thickness=1)
        # cv2.imshow('aa', image)
        # cv2.waitKey(0)
        # results_file.write(filename.split("/")[-1] +","+ str(int(box[1])) + " " + str(int(box[0])) +  " " + str(int(box[3])) +' ' +str(int(box[2]))+','+str(labels[i]) +','+str(scores[i])+ "\n")
    print("Predicting image: %s  "%filename, iii)
    iii += 1

    # print(os.path.join(save_path, basename[:-4]+'_pre'+basename[-4:]))
    cv2.imwrite(os.path.join(save_path, basename[:-4]+'_pre'+basename[-4:]),image)