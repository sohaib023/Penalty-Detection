import torch
import numpy as np
from truthpy import Rect

def seg_to_image(seg):
    seg = torch.argmax(seg, dim=0).cpu().detach().numpy()
    image = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    image[seg == 1] = [0, 255, 0]
    image[seg == 2] = [255, 255, 255]
    return image

def compute_contain(i: Rect, j: Rect):
    """compute the area ratio of smaller rectangle contained inside the other rectangle"""
    if i.area() == 0 or j.area() == 0:
        return 0.0
    return abs((i & j).area()) / min(i.area(), j.area())


def compute_IoU(i: Rect, j: Rect):
    """computes the overlap between two rectangles in the range (0, 1)"""
    if i.area() == 0 and j.area() == 0:
        return 0.0
    return abs((i & j).area()) / abs(i.area() + j.area() - (i & j).area())


def apply_nms(predictions, overlap_thresh=0.5, contain_thresh=0.9):
    """THIS METHOD PERFORMS THE NMS OPERATION ON ALL AVAILABLE PREDICTIONS,
    NMS = NON MAXIMA SUPRESSION"""
    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
    i = 0
    while i < len(predictions):
        j = i + 1
        while j < len(predictions):
            if (
                compute_contain(predictions[i][0], predictions[j][0]) > contain_thresh
                or compute_IoU(predictions[i][0], predictions[j][0]) > overlap_thresh
            ):
                rel_diff = 2 * (predictions[i][1] - predictions[j][1]) / (predictions[i][1] + predictions[j][1])
                if (rel_diff > 0.3):  
                    if predictions[i][2] != 'ball':
                        predictions.remove(predictions[j])
                        continue
            j += 1
        i += 1
    return predictions

def timestep_processing(segmentation, detections):
    to_remove = []
    for detection in detections:
        bbox = detection[0]
        x1,y1,x2,y2 = tuple(bbox)
        crop = segmentation[y1:y2, x1:x2]

        if detection[2] != 'ball':
            if (crop[:, :, 2] == 255).sum() / bbox.area() > 0.8:
                to_remove.append(detection)


class Tracker():
    def __init__(self):
        self.shot_taken = False
        self.frames_since_ball_missing = 0

        self.ball_pos = None
        self.ball_pos_pred = None
        self.ball_speed = None

    # def update(segmentation, detections):
    #     