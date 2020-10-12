from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse
import cv2

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator


def changeBGR2RGB(img):
    b = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    r = img[:, :, 2].copy()

    # RGB > BGR
    img[:, :, 0] = r
    img[:, :, 1] = g
    img[:, :, 2] = b

    return img


def changeRGB2BGR(img):
    r = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    b = img[:, :, 2].copy()

    # RGB > BGR
    img[:, :, 0] = b
    img[:, :, 1] = g
    img[:, :, 2] = r

    return img



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--video_path", type=str, default="data/video/video2.mp4", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3-custom.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/version2.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/custom/classes.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.2, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)

    # create output video directory
    os.makedirs("video-output", exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))
    model.eval()  # Set in evaluation mode
    classes = load_classes(opt.class_path)
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    #if opt.vedio_file.endswith(".mp4"):
    cap = cv2.VideoCapture(opt.video_path)
    colors = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")
    writer = None
    a=[]
    time_begin = time.time()
    NUM = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    FPS = int(cap.get(cv2.CAP_PROP_FPS))
    # Current frame
    cur_frame = 0
    # Output result
    file1 = open("Result.txt", "w+")
    

    #NUM=0
    while cap.isOpened():
        ret, img = cap.read()
        print("HI")
        if ret is False:
            break
        # img = cv2.resize(img, (1280, 960), interpolation=cv2.INTER_CUBIC)
        cur_frame += 1
        # txt wrote in file1
        txt = "current frame is {}, result is no move\n".format(cur_frame)
        
        
        RGBimg=changeBGR2RGB(img)
        imgTensor = transforms.ToTensor()(RGBimg)
        imgTensor, _ = pad_to_square(imgTensor, 0)
        imgTensor = resize(imgTensor, 416)
       
        imgTensor = imgTensor.unsqueeze(0)
        imgTensor = Variable(imgTensor.type(Tensor))
       

        with torch.no_grad():
            detections = model(imgTensor)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

        a.clear()
        if detections is not None:
            a.extend(detections)
        b=len(a)
        if len(a)  :
            for detections in a:
                if detections is not None:
                    detections = rescale_boxes(detections, opt.img_size, RGBimg.shape[:2])
                    unique_labels = detections[:, -1].cpu().unique()
                    n_cls_preds = len(unique_labels)
                    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                        box_w = x2 - x1
                        box_h = y2 - y1
                        color = [int(c) for c in colors[int(cls_pred)]]
                        
                        img = cv2.rectangle(img, (x1, y1 + box_h), (x2, y1), color, 2)
                        cv2.putText(img, classes[int(cls_pred)], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        cv2.putText(img, str("%.2f" % float(conf)), (x2, y2 - box_h), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    color, 2)

                        # frame classification    
                        if int(cls_pred) == 0:
                            txt = txt.replace("no move", "move1")
                        elif int(cls_pred) == 1:
                            txt = txt.replace("no move", "move2")

                                
        # write frame classification
        if(cur_frame % 10 == 0):
            file1.write(txt)    


        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter("video-output/detected-{}".format(opt.video_path.split("/")[2]),\
                                            fourcc,FPS,(img.shape[1],img.shape[0]),True)
        writer.write(changeRGB2BGR(RGBimg))


        cv2.imshow('frame', RGBimg)
        

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    time_end = time.time()
    time_total = time_end - time_begin

    print("Total frames: {}".format(NUM))
    print("Total time: {}".format(time_total))
    # print(NUM // time_total)

    cap.release()
    cv2.destroyAllWindows()
    writer.release()
    file1.close()
