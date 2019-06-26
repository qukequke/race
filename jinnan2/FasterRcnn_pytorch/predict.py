import os
import torch as t
from utils.config import opt
from model import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from data.util import  read_image
from utils.vis_tool import vis_bbox
from utils import array_tool as at
import glob
import cv2



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

trainer.load('/home/sar/AnnotatedNetworkModelGit-master/fasterrcnn/simple-faster-rcnn/fasterrcnn_03241017')

opt.caffe_pretrain=False# this model was trained from caffe-pretrained model

filenames = glob.glob('../test_data/*.jpg')

results_file = open("../submit/larger%s.csv"%str(0.5),"w")
iii = 1
# import matplotlib.pyplot as plt
for filename in filenames:
    img = read_image(filename)
    img = t.from_numpy(img)[None]
    _bboxes, _labels, _scores = trainer.faster_rcnn.predict(img,visualize=True)
    # print(_bboxes)
    bboxes = _bboxes[0]
    labels = _labels[0]
    scores = _scores[0]

    image = cv2.imread(filename)
    # fig, ax = plt.subplots(1,2,figsize=(5,10))
    for i,box in (enumerate(bboxes)):
        cv2.rectangle(image,(int(box[1]),int(box[0])),(int(box[3]),int(box[2])),color=(0,0,255),thickness=2 )
        cv2.putText(image,VOC_BBOX_LABEL_NAMES[labels[i]],(int(box[1]), int(box[0])),fontFace=cv2.FONT_HERSHEY_COMPLEX,fontScale=0.5,color=(0,0,255), thickness=1)
        # cv2.imshow('aa', image)
        # cv2.waitKey(0)
        results_file.write(filename.split("/")[-1] +","+ str(int(box[1])) + " " + str(int(box[0])) +  " " + str(int(box[3])) +' ' +str(int(box[2]))+','+str(labels[i]) +','+str(scores[i])+ "\n")
    print("Predicting image: %s  "%filename, iii)
    iii += 1

    cv2.imwrite("../outputs1/%s"%filename.split("/")[-1],image)

# vis_bbox(at.tonumpy(img[0]),
#          at.tonumpy(_bboxes[0]),
#          at.tonumpy(_labels[0]).reshape(-1),
#          at.tonumpy(_scores[0]).reshape(-1))

# import numpy as np
# import os
#
# import cv2
# import skimage
# import torch
# from dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
# from torchvision import transforms as T
# from glob import glob
# assert torch.__version__.split('.')[1] == '4'
#
# print('CUDA available: {}'.format(torch.cuda.is_available()))
#
# # threshold for class score
# threshold = 0.5
# results_file = open("./submit/larger%s.csv"%str(threshold),"w")
#
# if not os.path.exists("./submit/"):
#     os.mkdir("./submit/")
#
# if not os.path.exists("./outputs/"):
#     os.mkdir("./outputs/")
# # if not os.path.exists("./best_models/"):
#     # os.mkdir("./best_models/")
#
# def demo(image_lists):
#     classes = ["gangjin"]
#     cur_model = "./models/model_final.pth"
#     print('use model', cur_model)
#     # cur_model = "./models/8_scale15_0.9489300025577628.pth"
#     # cur_model = "./models/20_scale15_0.9579579579579579.pth"
#     # para = torch.load(cur_model).module.state_dict()
#     # retinanet = model.resnet152(num_classes=5,pretrained=False)
#     # retinanet = model.resnet18(5)
#     retinanet= torch.load(cur_model)
#     # retinanet = retinanet.load_state_dict(para)
#
#     retinanet = retinanet.cuda()
#     retinanet.eval()
#     #detect
#     transforms = T.Compose([
#         Normalizer(),
#         Resizer()
#         ])
#     iii = 0
#     for filename in image_lists:
#         image = skimage.io.imread(filename)
#         sampler = {"img":image.astype(np.float32)/255.0,"annot":np.empty(shape=(5,5))}
#         image_tf = transforms(sampler)
#         scale = image_tf["scale"]
#         new_shape = image_tf['img'].shape
#         x = torch.autograd.Variable(image_tf['img'].unsqueeze(0).transpose(1,3), volatile=True)
#         with torch.no_grad():
#             scores,labels,bboxes = retinanet(x.cuda().float())
#             bboxes /= scale
#             scores = scores.cpu().data.numpy()
#             bboxes = bboxes.cpu().data.numpy()
#             labels = labels.cpu().data.numpy()
#             # select threshold
#             idxs = np.where(scores > threshold)[0]
#             scores = scores[idxs]
#             bboxes = bboxes[idxs]
#             labeles = labels[idxs]
#             #embed()
#             for i,box in (enumerate(bboxes)):
#                  cv2.rectangle(image,(int(box[1]),int(box[0])),(int(box[3]),int(box[2])),color=(0,0,255),thickness=2 )
#                  results_file.write(filename.split("/")[-1] +","+ str(int(box[1])) + " " + str(int(box[0])) +  " " + str(int(box[3])) +' ' +str(int(box[2]))+','+str(labeles[i]) +','+str(scores[i])+ "\n")
#             print("Predicting image: %s  "%filename, iii)
#             iii += 1
#
#             cv2.imwrite("./outputs/%s"%filename.split("/")[-1],image)
# if __name__ == "__main__":
#     root = "./test_data/"
#     image_lists = glob(root+"*.jpg")
#     demo(image_lists)
