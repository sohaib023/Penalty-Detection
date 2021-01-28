import os
import glob
import math
import pickle
import argparse

import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import torch
import torchvision.transforms.functional as TF

from truthpy import Rect

from libs import utils
from libs.MultiTaskModel import MultiTaskModel

INPUT_SHAPE = [750, 1333]
CLASSES = ['player', 'referee', 'ball']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEG_PREV = None
ALPHA = 0.4

def infer(model, frame):
    global SEG_PREV
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    org_w, org_h = pil_image.size[:2]
    
    image = TF.resize(pil_image, INPUT_SHAPE, Image.BILINEAR)
    image = TF.to_tensor(image).to(device)
    
    output_dict = model([image])[0]
    
    h, w = image.shape[-2:]
    rw = org_w/w
    rh = org_h/h

    ######################## PROCESSING DETECTIONS ###########################

    detections = []
    for box, score, label in zip(output_dict['boxes'], output_dict['scores'], output_dict['labels']):
        if score < 0.2:
            continue

        x, y, x2, y2 = box
        detections.append([Rect(int(x * rw), int(y * rh), int(x2 * rw), int(y2 * rh)), score.item(), CLASSES[label]])

    ############################### PROCESSING SEGMENTATION MASK ############################

    seg = output_dict['seg']
    seg = torch.nn.functional.interpolate(seg[None], size=[org_h, org_w], mode='nearest')[0]
    seg = seg.cpu().detach().numpy()
    if SEG_PREV is not None:
        seg = ALPHA * seg + (1-ALPHA) * SEG_PREV
    SEG_PREV = seg

    seg_image = utils.seg_to_image(seg)
    return detections, seg_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-i',
        '--input_video',
        type=str,
        help="Path to directory containing input dataset.",
        default="data/videos_from_left/v8 left.mp4"
    )
    parser.add_argument(
        '-o',
        '--out_path',
        type=str,
        help="Path for outputting model weights and tensorboard summary.",
        default="data/videos_out_final"
    )
    parser.add_argument(
        '-c',
        '--checkpoint',
        type=str,
        help="Path to checkpoint file to be used for inference.",
        default="output/epoch_35.pth"
    )

    args = parser.parse_args()

    os.makedirs(args.out_path, exist_ok=True)
    
    model = MultiTaskModel(training=False)
    model.to(device)
    model.eval()

    ckpt = torch.load(args.checkpoint)
    model.load_state_dict(ckpt['model_state_dict'])

    cap = cv2.VideoCapture(args.input_video)

    if not cap.isOpened():
        print("Cannot open video file")
        exit(0)

    fps = round(cap.get(cv2.CAP_PROP_FPS))
    w   = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(os.path.join(args.out_path, os.path.basename(args.input_video) + '-seg.mp4'), cv2.VideoWriter_fourcc('M','J','P','G'), fps, (w, h))
    out_det = cv2.VideoWriter(os.path.join(args.out_path, os.path.basename(args.input_video) + '-det.mp4'), cv2.VideoWriter_fourcc('M','J','P','G'), fps, (w, h))

    video_detections = []
    frame_num = 0
    tracker = utils.Tracker()

    FRAME_PREV = None
    while True:
        ret, frame = cap.read()
        print(frame_num)
        if not ret:
            print("EOF reached. Exiting ...")
            break
        # frame = cv2.resize(frame, (1920, frame.shape[0] * 1920 // frame.shape[1]), interpolation=cv2.INTER_LINEAR)

        if FRAME_PREV is None:
            frame_num += 1
            FRAME_PREV = frame
            continue

        with torch.no_grad():
            detections, seg_image = infer(model, frame)

        ######### POST-PROCESSING #########
        image = frame.copy()

        detections = utils.apply_nms(detections)

        seg_image, detections = utils.timestep_processing(frame, seg_image, detections)
        if (frame - FRAME_PREV).mean() > 20:
            tracker.update(frame, seg_image, detections)

        tracker.visualize(image)

        ############################# OUTPUTTING FRAMES AND DATA ################################
        utils.disp_img(image, "detections")
        cv2.waitKey(1)

        # out_det.write(cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA))
        # out.write(cv2.resize(seg_image, (w, h), interpolation=cv2.INTER_AREA))
        out_det.write(image)
        out.write(seg_image)
        
        video_detections.append(detections)
        
        frame_num += 1
        FRAME_PREV = frame
        
    # with open(os.path.join(args.out_path, os.path.basename(args.input_video).rsplit('.', 1)[0] + '.pkl'), "wb") as f:
    #     pickle.dump(video_detections, f)

    # plt.plot(range(len(tracker.goal_signals_ang)), tracker.goal_signals_ang)
    # plt.savefig(os.path.join(args.out_path, os.path.basename(args.input_video).rsplit('.', 1)[0] + '-ang.png'))
    # plt.clf()
    # plt.plot(range(len(tracker.goal_signals_ang)), tracker.goal_signals_trans)
    # plt.savefig(os.path.join(args.out_path, os.path.basename(args.input_video).rsplit('.', 1)[0] + '-trans.png'))
    # plt.clf()
    # plt.plot(range(len(tracker.goal_signals_ang)), tracker.goal_signals_mean)
    # plt.savefig(os.path.join(args.out_path, os.path.basename(args.input_video).rsplit('.', 1)[0] + '-mean.png'))
    # plt.clf()
    
    cap.release()
    out.release()
    out_det.release()