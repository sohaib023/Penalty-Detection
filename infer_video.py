import os
import glob
import pickle
import argparse

import cv2
import numpy as np
from PIL import Image

import torch
import torchvision.transforms.functional as TF

from truthpy import Rect

from libs import utils
from libs.MultiTaskModel import MultiTaskModel

INPUT_SHAPE = [750, 1333]
CLASSES = ['player', 'referee', 'ball']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def seg_to_image(seg):
    seg = torch.argmax(seg, dim=0).cpu().detach().numpy()
    image = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    image[seg == 0] += np.array([0, 0, 255], dtype=np.uint8)[None, :]
    image[seg == 1] += np.array([0, 255, 0], dtype=np.uint8)[None, :]
    image[seg == 2] += np.array([255, 0, 0], dtype=np.uint8)[None, :]
    return image

def infer(model, frame):
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
        # if label != CLASSES.index('ball'):
        #     if score < 0.2:
        #         continue
        # elif score < 0.5:
        #     continue

        x, y, x2, y2 = box
        detections.append((Rect(int(x * rw), int(y * rh), int(x2 * rw), int(y2 * rh)), score.item(), CLASSES[label]))

    ############################### PROCESSING SEGMENTATION MASK ############################

    seg = output_dict['seg']
    seg = torch.nn.functional.interpolate(seg[None], size=[org_h, org_w], mode='nearest')[0]
    seg_image = seg_to_image(seg)
    return detections, seg_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-i',
        '--input_video',
        type=str,
        help="Path to directory containing input dataset.",
        default="data/videos_from_left/vleft21.mp4"
    )
    parser.add_argument(
        '-o',
        '--out_path',
        type=str,
        help="Path for outputting model weights and tensorboard summary.",
        default="data/videos_out"
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

    out = cv2.VideoWriter(os.path.join(args.out_path, os.path.basename(args.input_video)), cv2.VideoWriter_fourcc('M','J','P','G'), fps, (w, h))
    out_det = cv2.VideoWriter(os.path.join(args.out_path, os.path.basename(args.input_video) + '-det.mp4'), cv2.VideoWriter_fourcc('M','J','P','G'), fps, (w, h))

    video_detections = []
    frame_num = 0

    while True:
        ret, frame = cap.read()
        print(frame_num)
        if not ret:
            print("EOF reached. Exiting ...")
            break

        detections, seg_image = infer(model, frame)

        ######### POST-PROCESSING #########

        detections = utils.apply_nms(detections)
        utils.timestep_processing(seg_image, detections)
        # utils.temporal_processing(seg_image, detections, state)

        ########## VISUALIZATION ##########
        image = frame.copy()
        
        for detection in detections:
            x, y, x2, y2 = tuple(detection[0])
            score, label = detection[1:]

            cv2.rectangle(image, (x, y), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image, '{:.2f}'.format(score), (x, y), 
                cv2.FONT_HERSHEY_SIMPLEX,  
                1, (255, 0, 0), 2, cv2.LINE_AA
            ) 
            cv2.putText(
                image, label, (x, y2 +20), 
                cv2.FONT_HERSHEY_SIMPLEX,  
                1, (255, 0, 0), 2, cv2.LINE_AA
            ) 

        ############################# OUTPUTTING FRAMES AND DATA ################################
        out_det.write(image)
        video_detections.append(detections)

        out.write(seg_image)
        frame_num += 1

    with open(os.path.join(args.out_path, os.path.basename(args.input_video).rsplit('.', 1)[0] + '.pkl'), "wb") as f:
        pickle.dump(video_detections, f)
    cap.release()
    out.release()
    out_det.release()