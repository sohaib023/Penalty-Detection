import os
import time
import math
import cv2
import numpy as np
from scipy.stats import mode
from libs import utils

def calc_angl_n_transl(flow, step=20):
    
    '''
    input:
        - img - numpy array - image
        - flow - numpy array - optical flow
        - step - int - measurement of sparsity
    output:
        - angles - numpy array - array of angles of optical flow lines to the x-axis
        - translation - numpy array - array of length values for optical flow lines
        - lines - list - list of actual optical flow lines (where each line represents a trajectory of 
        a particular point in the image)
    '''

    angles = []
    translation = []

    h, w = flow.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx*3, y+fy*3]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    
    for (x1, y1), (x2, y2) in lines:
        angle = math.atan2(- int(y2) + int(y1), int(x2) - int(x1)) * 180.0 / np.pi
        length = math.hypot(int(x2) - int(x1), - int(y2) + int(y1))
        translation.append(length)
        angles.append(angle)
    
    return np.array(angles), np.array(translation), lines

def estimate_motion(angles, translation):
    
    '''
    Input:
        - angles - numpy array - array of angles of optical flow lines to the x-axis
        - translation - numpy array - array of length values for optical flow lines
    Output:
        - ang_mode - float - mode of angles of trajectories. can be used to determine the direction of movement
        - transl_mode - float - mode of translation values 
        - ratio - float - shows how different values of translation are across a pair of frames. allows to 
        conclude about the type of movement
        - steady - bool - show if there is almost no movement on the video at the moment
    '''
    
    # Get indices of nonzero opical flow values. We'll use just them
    nonzero = np.where(translation > 0)
    
    # Whether non-zero value is close to zero or not. Should be set as a thershold
    # steady = np.mean(translation) < 0.5
    
    translation = translation[nonzero]
    if len(translation) < 2:
        return 0, 0, 0, True, 0

    transl_std = np.std(translation)
    mean = np.mean(translation)
    transl_mode = mode(translation)[0][0]
    
    angles = angles[nonzero]
    ang_std = np.std(angles)
    ang_mode = mode(angles)[0][0]
    
    # cutt off twenty percent of the sorted list from both sides to get rid off outliers
    ten_percent = len(translation) // 10
    translations = sorted(translation)
    translations = translations[ten_percent: len(translations) - ten_percent]
    
    print(transl_std, ang_std)
    return ang_mode, transl_mode, ang_std, transl_std, mean

def draw_flow(img, lines):
    
    '''
    input:
        - img - numpy array - image to draw on
        - lines - list - list of lines to draw
        - BGR image with visualised optical flow
    '''

    width_delay_ratio = 6
    height_delay_ratio = 5
    
    h, w = img.shape[:2]
        
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)

    return vis

def optical_flow_pipeline(frame, FRAME_PREV, bbox_goal, bbox_gk):
            # frame = frame[:h//2, :w//2]
    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # hsv[hsv[:, :, 1] > 100, 2] = 0

    # hsv[hsv[:, :, 2] > 125, 2] = 255
    # hsv[hsv[:, :, 2] < 125, 2] = 0

    # hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # utils.disp_img(frame, "frame")
    # utils.disp_img(hsv, "hsv")

    x, y, x2, y2 = bbox_goal
    # frame = cv2.resize(frame[y:y2, x:x2], (800, 450), interpolation=cv2.INTER_AREA)
    # FRAME_PREV = cv2.resize(FRAME_PREV[y:y2, x:x2], (800, 450), interpolation=cv2.INTER_AREA)
    frame = frame[y:y2, x:x2]
    FRAME_PREV = FRAME_PREV[y:y2, x:x2]

    rx = frame.shape[1] / (x2 - x)
    ry = frame.shape[0] / (y2 - y)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    FRAME_PREV = cv2.cvtColor(FRAME_PREV, cv2.COLOR_BGR2GRAY)

    kernel = np.array([[1,1,1], [1,1,1], [1,1,1]])
    kernel = kernel / np.sum(kernel)
    frame = cv2.filter2D(frame, -1, kernel)
    FRAME_PREV = cv2.filter2D(FRAME_PREV, -1, kernel)

    transl_std = 0
    ang_std = 0
    mean = 0
    if FRAME_PREV is not None:
        flow = cv2.calcOpticalFlowFarneback(FRAME_PREV, frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        xg, yg, _, _ = bbox_goal
        x, y, x2, y2 = bbox_gk
        x -= xg
        y -= yg
        x2 -= xg
        y2 -= yg
        x   = int(x * rx)
        x2  = int(x2 * rx)
        y   = int(y * ry)
        y2  = int(y2 * ry)
        flow[max(y - 20, 0): y2 + 20, max(x - 20, 0): x2 + 20] = 0

        # calculate trajectories and analyse them
        angles, transl, lines = calc_angl_n_transl(flow)
        ang_mode, transl_mode, ang_std, transl_std, mean = estimate_motion(angles, transl)

        # draw trajectories on the frame
        vis = draw_flow(frame.copy(), lines)
        # next_gray = cv2.cvtColor(next_gray.copy(), cv2.COLOR_GRAY2BGR)
        utils.disp_img(vis, "flow")

    return transl_std, ang_std, mean