import math

import cv2
import torch
import numpy as np
from truthpy import Rect

from scipy.optimize import linear_sum_assignment

from libs.optical import optical_flow_pipeline

CROWD_THRESH_REMOVE = 0.8

def disp_img(image, label="default"):
    h = 700
    resized = cv2.resize(image, (image.shape[1] * h // image.shape[0], h), cv2.INTER_NEAREST)
    cv2.imshow(label, resized)

def seg_to_image(seg):
    seg = np.argmax(seg, axis=0)
    image = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    image[seg == 0] += np.array([0, 0, 255], dtype=np.uint8)[None, :]
    image[seg == 1] += np.array([0, 255, 0], dtype=np.uint8)[None, :]
    image[seg == 2] += np.array([255, 0, 0], dtype=np.uint8)[None, :]
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


def apply_nms(predictions, overlap_thresh=0.7, contain_thresh=0.7):
    """THIS METHOD PERFORMS THE NMS OPERATION ON ALL AVAILABLE PREDICTIONS,
    NMS = NON MAXIMA SUPRESSION"""
    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
    i = 0
    while i < len(predictions):
        j = i + 1
        while j < len(predictions):
            contain = compute_contain(predictions[i][0], predictions[j][0])
            iou = compute_IoU(predictions[i][0], predictions[j][0])
            if (
                contain > contain_thresh
                or iou > overlap_thresh
            ):
                rel_diff = 2 * (predictions[i][1] - predictions[j][1]) / (predictions[i][1] + predictions[j][1])
                if (rel_diff > 0.01 or iou > 0.85):  
                    print("Removing nms", predictions[i])
                    if predictions[j][2] != 'ball':
                        predictions.remove(predictions[j])
                        continue
            j += 1
        i += 1
    return predictions

def remove_extra_ground_blobs(segmentation):
    # Retain the largest ground blob towards bottom side of image and remove the rest.
    h, w = segmentation.shape[:2]
    G = segmentation[:, :, 1]
    inv_G = ~G
    cv2.floodFill(inv_G, None, (w//2, h - 50), 255)
    G = (G & inv_G)
    segmentation[:, :, 1] = G
    segmentation[segmentation.sum(axis=2) == 0] = [255, 0, 0]
    return segmentation

def remove_extra_goal_blobs(segmentation):
    # Retain the largest ground blob towards bottom side of image and remove the rest.
    h, w = segmentation.shape[:2]
    B = segmentation[:, :, 0]
    contours, hierarchy = cv2.findContours(B, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return segmentation

    max_cnt = max(contours, key=lambda cnt: cv2.contourArea(cnt))
    B[:, :] = 0
    
    B = cv2.drawContours(np.array(B), [max_cnt], 0, 255, -1)

    segmentation[:, :, 0] = B
    # segmentation[segmentation.sum(axis=2) == 0] = [255, 0, 0]
    return segmentation

# def goal_smoothing(segmentation):
#     B = segmentation[:, :, 0]
#     contours, hierarchy = cv2.findContours(B, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

#     mask = np.zeros(B.shape, np.uint8)

#     areas = [cv2.contourArea(cnt) for cnt in contours]
#     avg = sum(areas) / len(areas)
#     contours = [cnt for cnt, area in zip(contours, areas) if area > avg *0.5]

#     print(type(contours[0]))
#     cnt = np.array([x for cnt in contours for x in cnt])
#     print(type(cnt))
#     perimeter = cv2.arcLength(cnt,True)
#     epsilon = 0.01*cv2.arcLength(cnt,True)
#     cv2.drawContours(mask, cnt, -1, 255, -1)
#     approx = cv2.approxPolyDP(cnt,epsilon,True)

#     mask = np.zeros(B.shape, np.uint8)
#     cv2.drawContours(mask, cnt, -1, 255, 3)
#     disp_img(mask, "cnt")
#     mask = np.zeros(B.shape, np.uint8)
#     cv2.drawContours(mask, [approx], -1, 255, 3)
#     disp_img(mask, "approx")

def find_referees(image, detections):
    histograms = []
    for det in detections:
        b = det[0]
        crop = image[b.y1:b.y2, b.x1:b.x2]
        hist = cv2.calcHist([crop], [0, 1, 2], None, [12, 12, 12],
            [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        histograms.append(hist)

    correlation_map = [[0]* len(detections) for i in range(len(detections))]
    max_corr = 0
    indices = (-1, -1)
    for i, hist in enumerate(histograms):
        if detections[i][2] in ['player', 'referee']:
            for j, hist2 in enumerate(histograms):
                if i != j and detections[j][2] in ['player', 'referee']:
                    correlation_map[i][j] = cv2.compareHist(hist, hist2, cv2.HISTCMP_CORREL)
                    if max_corr < correlation_map[i][j]:
                        max_corr = correlation_map[i][j]
                        indices = (i, j)

    # detections[indices[0]][2] = 'ref confirmed'
    # detections[indices[1]][2] = 'ref confirmed'


def timestep_processing(image, segmentation, detections):
    h, w = segmentation.shape[:2]
    segmentation = remove_extra_goal_blobs(segmentation)
    # goal_smoothing(segmentation)

    kernel = np.ones((35, 15), dtype=np.uint8)
    is_red = segmentation[0, :, 2] == 255
    segmentation[:, :, 0] = cv2.dilate(segmentation[:, :, 0], kernel, iterations=3)
    segmentation[0, is_red, 0] = 0
    segmentation[:, :, 0] = cv2.erode(segmentation[:, :, 0], kernel, iterations=3)
    segmentation[segmentation[:, :, 0] == 255] = [255, 0, 0]

    to_remove = []
    person_sizes = []

    for detection in detections:
        bbox = detection[0]
        x1,y1,x2,y2 = tuple(bbox)
        crop = segmentation[y1:y2, x1:x2]

        if detection[2] != 'ball':
            if (crop[:, :, 2] == 255).sum() / bbox.area() > CROWD_THRESH_REMOVE or \
                    bbox.area() > h*w*0.2:
                to_remove.append(detection)
            else:
                person_sizes.append(bbox.area())

    if len(person_sizes) > 0:
        avg_size = sum(person_sizes) / len(person_sizes)

        for detection in detections:
            bbox = detection[0]
            x1,y1,x2,y2 = tuple(bbox)
            crop = segmentation[y1:y2, x1:x2]

            if detection[2] != 'ball' and (bbox.area() < avg_size * 0.5 or bbox.area() > avg_size * 3):
                to_remove.append(detection)

    # print("Removed detections = {}".format(len(to_remove)))

    detections = [det for det in detections if det not in to_remove]
    
    # find_referees(image, detections)
    
    return segmentation, detections

ALPHA = 0.5

class Object():
    def __init__(self, coords, velocity, label, conf):
        self.x, self.y, self.w, self.h = coords
        self.vx, self.vy = velocity

        self.labels = [label]
        self.hits = 0
        self.isactive = True
        self.conf = conf
        self.age = 0
        self.missed_frames = 0

    def get_future(self):
        other = Object((self.x, self.y, self.w, self.h), (self.vx, self.vy), self.labels[-1], self.conf)
        other.x = int(other.x + self.vx)
        other.y = int(other.y + self.vy)
        return other

    def update_position(self, new_pos, shape):
        self.vx = (new_pos[0] - self.x) * ALPHA + (1 - ALPHA) * self.vx
        self.vy = (new_pos[1] - self.y) * ALPHA + (1 - ALPHA) * self.vy

        self.w = int(shape[0] * ALPHA + (1 - ALPHA) * self.w)
        self.h = int(shape[1] * ALPHA + (1 - ALPHA) * self.h)

        self.x, self.y = new_pos
        self.missed_frames = 0

    def get_distance(self, other):
        return np.linalg.norm([self.x - other.x, self.y - other.y])

    def to_rect(self):
        return Rect(self.x - self.w // 2, self.y - self.h // 2, self.x + self.w // 2, self.y + self.h // 2)

class Tracker():
    def __init__(self):
        self.frame_counter = 0

        self.objects = {}
        # self.gk = None
        # self.pk = None

        # self.refs = [None, None, None]

        # self.ball = None

        # self.objects = []
        # self.blacklisted = []

        # self.shot_taken = False
        self.frames_since_ball_missing = 0

        # self.ball_seed_found = False

        self.h, self.w = None, None

        self.goal_side = None
        self.goal_coords = None
        self.goal_scored_votes = 0
        self.goal_scored = False
        self.goal_coords_frame0 = None

        self.state = 0

        self.state_descriptions = [
            "Run-up",
            "Taking Shot",
            "Tracking Ball",
        ]

        self.prev_frame = None
        self.unmatched_objects = []

        self.goal_signals_trans = []
        self.goal_signals_ang = []
        self.goal_signals_mean = []

    def search_ball(self, segmentation, objects):
        candidates = []
        for obj in objects:
            if obj.labels[-1] == 'ball':
                if self.state == 0:
                    if obj.y > self.h * 0.4:
                        if self.goal_side == 'L':
                            if obj.x > self.w // 2:
                                candidates.append(obj)
                        elif obj.x < self.w // 2:
                            candidates.append(obj)
                elif self.state == 2:
                    if self.goal_side == 'L':
                        if obj.x < self.w // 2:
                            candidates.append(obj)
                    elif obj.x < self.w // 2:
                        candidates.append(obj)

        if len(candidates) > 0:
            ball = max(candidates, key=lambda c: c.conf)
            if ball.conf > 0.5 or self.frame_counter > 10:
                self.objects['ball'] = ball
                objects.remove(ball)

    def search_gk(self, segmentation, objects):
        candidates = []
        for obj in objects:
            if obj.labels[-1] in ['player', 'referee']:
                # if self.goal_side == 'L':
                    if self.goal_coords[0] - 100 < obj.x < self.goal_coords[2] + 100:
                        candidates.append(obj)
                # elif self.goal_coords[0] - 100 > obj.x > self.goal_coords[2] + 100:
                    # candidates.append(obj)
        # print(candidates)

        if len(candidates) > 0:
            ratios=[]
            for c in candidates:
                rect = c.to_rect()
                x1, y1, x2, y2 = tuple(rect)
                ratios.append(segmentation[y1:y2, x1:x2, 0].sum() / (255 * rect.area()))

            gk = max(zip(candidates, ratios), key=lambda c: c[1])
            if (gk[0].conf > 0.5 and gk[1] > 0.5) or self.frame_counter > 10:
                self.objects['gk'] = gk[0]
                objects.remove(gk[0])

    def search_pk(self, segmentation, objects):
        if 'ball' not in self.objects and self.frame_counter < 10:
            return

        if 'ball' in self.objects:
            ball_x, ball_y = self.objects['ball'].x, self.objects['ball'].y
        else:
            if self.goal_side == 'L':
                ball_x = (self.w * 0.6)
            else:
                ball_x = (self.w * 0.4)
            ball_y = self.h // 2

        candidates = []
        for obj in objects:
            if obj.labels[-1] in ['player', 'referee']:
                if self.goal_side == 'L':
                    if obj.x > ball_x:
                        candidates.append(obj)
                elif obj.x < ball_x:
                    candidates.append(obj)

        if len(candidates) > 0:
            distances = []
            for c in candidates:
                x, y = c.x, c.y + c.h // 2
                distance = np.linalg.norm([x - ball_x, 3 * (y - ball_y)]) / (self.w / 3)
                distances.append(distance)

            pk = max(zip(candidates, distances), key=lambda c: -c[1] + c[0].conf + (0.2 if c[0].labels[-1] == 'player' else 0))[0]
            if (pk.conf > 0.5 and abs(ball_y - pk.y) < self.h//3) or self.frame_counter > 10:
                self.objects['pk'] = pk
                objects.remove(pk)

    def update_key_objects(self, image, segmentation, detections):
        w = self.w
        to_remove = []

        if len(detections) == 0:
            return

        for i, (name, obj) in enumerate(self.objects.items()):
            future_obj = obj.get_future()
            distances = []
            for j, newobj in enumerate(detections):
                # One of them has to be true
                distance = 100000
                if name == 'ball':
                    if newobj.labels[-1] == 'ball':
                        if self.state == 0 or self.state == 2:
                            distance = future_obj.get_distance(newobj)
                        elif self.state == 1:
                            pk = self.objects['pk']
                            if self.goal_side == 'L':
                                if newobj.x < min(pk.x - pk.w, obj.x):
                                    distance = future_obj.get_distance(newobj)
                            else:
                                if newobj.x > max(pk.x + pk.w, obj.x):
                                    distance = future_obj.get_distance(newobj)
                else:
                    if newobj.labels[-1] != 'ball':
                        distance = future_obj.get_distance(newobj)
                distances.append(distance)

            min_dist = min(distances)
            is_accepted = False
            if name == 'ball':
                if self.state == 0 and min_dist < (future_obj.w + future_obj.h):
                    is_accepted = True
                elif (self.state == 1 or self.state == 2) and min_dist < np.linalg.norm([obj.vx, obj.vy]) * 3:
                    is_accepted = True
            else:
                if min_dist < (future_obj.w + future_obj.h) // 2:
                    is_accepted = True

            if is_accepted:
                newobj = detections[distances.index(min_dist)]
                to_remove.append(newobj)
                new_pos = newobj.x, newobj.y
                shape = newobj.w, newobj.h
                obj.update_position(new_pos, shape)

                obj.labels.append(newobj.labels[-1])
                obj.labels = obj.labels[-10:]

        for x in to_remove:
            if x in detections:
                detections.remove(x)


        # similarity_matrix = np.zeros((len(self.objects.keys()), len(detections)))
        # for i, (name, obj) in enumerate(self.objects.items()):
        #     for j, newobj in enumerate(detections):
        #         score = 0
        #         distance = 1e-4 + obj.get_future().get_distance(newobj)
        #         size_diff = 1e-4 + np.linalg.norm([obj.w - newobj.w, obj.h - newobj.h], ord=1)

        #         div_factor = 5
        #         # print(obj.w, obj.h)
        #         # if name == 'ball':
        #             # div_factor = 1
        #             # print(distance)
        #         if distance < np.linalg.norm([abs(obj.vx) + w // div_factor, abs(obj.vy) + w // div_factor]):
        #             div_factor *= 2
        #             score += np.linalg.norm([abs(obj.vx) + w // div_factor, abs(obj.vy) + w // div_factor]) / distance
        #             score += (obj.w + obj.h) / (size_diff * 2)

        #             if name =='ball' and newobj.labels[-1] != 'ball':
        #                 continue
        #             elif name !='ball' and newobj.labels[-1] == 'ball':
        #                 continue
        #             # else:
        #                 # score += ((obj.labels[-5:].count(newobj.labels[-1])) / len(obj.labels[-5:]))
        #         similarity_matrix[i][j] = score

        # row_ind, col_ind = linear_sum_assignment(similarity_matrix, maximize=True)

        # keys = list(self.objects.keys())
        # for idx_r, idx_c in zip(row_ind, col_ind):
        #     name = keys[idx_r]
        #     obj = self.objects[name]
        #     new_obj = detections[idx_c]

        #     if obj.age == 0:
        #         continue
        #     print(name, np.round(similarity_matrix[idx_r], 2))
        #     if name == 'ball':
        #         if similarity_matrix[idx_r, idx_c] <= 10 / obj.missed_frames:
        #             continue
        #     elif similarity_matrix[idx_r, idx_c] <= 200 / obj.missed_frames:
        #         continue
        #     new_pos = new_obj.x, new_obj.y
        #     shape = new_obj.w, new_obj.h
        #     obj.update_position(new_pos, shape)

        #     obj.labels.append(new_obj.labels[-1])
        #     obj.labels = obj.labels[-10:]

    def goal_detection(self, image, segmentation, detections):
        # indices = np.where(segmentation[:, :, 0] == 255)
        if segmentation[:, :, 0].sum() == 0:
            return False

        # if len(indices[0]) == 0:
        #     return False
        
        # self.goal_coords = [min(indices[1]), min(indices[0]), max(indices[1]) + 1, max(indices[0]) + 1]

        area = (self.goal_coords[3] - self.goal_coords[1]) * (self.goal_coords[2] - self.goal_coords[0])
        blue_pixels = segmentation[:, :, 0].sum() // 255
        x, y, x2, y2 = self.goal_coords_frame0
        if area < 0.4 * (y2 - y) * (x2 - x) or blue_pixels < area * 0.3:
            return False

        if self.prev_frame is None:
            self.prev_frame = image
            return False

        bbox_gk = [0, 0, 0, 0]
        if 'gk' in self.objects:
            bbox_gk = list(self.objects['gk'].to_rect())

        trans, ang, trans_mean = optical_flow_pipeline(image, self.prev_frame, self.goal_coords, bbox_gk)
        self.goal_signals_ang.append(ang)
        self.goal_signals_trans.append(trans)
        self.goal_signals_mean.append(trans_mean)
        
        self.prev_frame = image
        if trans > 5 and ang > 40:
            score = trans * 6 + ang
            if score > 80:
                # print("Goal scored")
                return True
        return False

    def update(self, image, segmentation, detections):
        try:
            self.h, self.w = segmentation.shape[:2]
            indices = np.where(segmentation[:, :, 0] == 255)
            if len(indices[0]) != 0:
                self.goal_coords = [min(indices[1]), min(indices[0]), max(indices[1]), max(indices[0])]
                if self.goal_coords_frame0 is None:
                    self.goal_coords_frame0 = self.goal_coords
            else:
                return

            if self.goal_side is None:
                if segmentation[:, :self.w // 2, 0].sum() > segmentation[:, self.w // 2:, 0].sum():
                    self.goal_side = 'L'
                else:
                    self.goal_side = 'R'

            curr_objects = []
            for d in detections:
                b = d[0]
                curr_objects.append(
                    Object(
                        ((b.x1 + b.x2) // 2, (b.y1 + b.y2) // 2, b.w, b.h),
                        (0, 0),
                        d[2],
                        d[1]
                    )
                )

            for key, obj in self.objects.items():
                obj.missed_frames += 1

            if 'ball' not in self.objects or self.objects['ball'].missed_frames > (10 if self.state == 1 else 5):
                self.search_ball(segmentation, curr_objects)
            if 'gk' not in self.objects or self.objects['gk'].missed_frames > 5:
                self.search_gk(segmentation, curr_objects)
            if 'pk' not in self.objects or self.objects['pk'].missed_frames > 5:
                self.search_pk(segmentation, curr_objects)

            if self.state == 0 and 'ball' in self.objects and 'pk' in self.objects:
                ball = self.objects['ball'].to_rect()
                pk = self.objects['pk'].to_rect()
                random_obj_overlap = False
                for obj in curr_objects:
                    if obj.labels[-1] in ['player', 'referee']:
                        if (ball & obj.to_rect()).area() / ball.area() >= 0.8:
                            random_obj_overlap = True
                            break

                pk.w += self.w//15
                if (ball & pk).area() / ball.area() >= 0.5 or \
                    (self.goal_side == 'L' and ball.x1 < self.w // 2) or \
                    (self.goal_side == 'R' and ball.x2 > self.w // 2) or \
                    random_obj_overlap:
                    self.state = 1
                    ball = self.objects['ball']
                    gk = self.objects['gk']
                    ball.vx = (gk.x - ball.x) / 8
                    ball.vy = (gk.y - ball.y) / 8
                pk.w -= self.w//15

            if self.state == 1 and 'ball' in self.objects:
                ball = self.objects['ball']
                if (ball.x > self.goal_coords[0] - 200 and self.goal_side == 'R') or \
                   (ball.x < self.goal_coords[2] + 200 and self.goal_side == 'L'):
                        self.state = 2

            if self.state == 2:
                self.goal_scored_votes += int(self.goal_detection(image, segmentation, detections))
                self.goal_scored_votes *= 0.9
                if self.goal_scored_votes > 2:
                    self.goal_scored = True
            else:
                self.goal_signals_ang.append(0)
                self.goal_signals_trans.append(0)
                self.goal_signals_mean.append(0)

            self.update_key_objects(image, segmentation, curr_objects)
            
            self.unmatched_objects = curr_objects
        except Exception as e:
            pass

        for key, obj in self.objects.items():
            obj.age += 1

        self.frame_counter += 1

    def visualize(self, image):
        text = "Analyzing..."
        if self.goal_scored:
            text = "Goal Scored!"
        cv2.putText(
            image, text, (10, 40), 
            cv2.FONT_HERSHEY_SIMPLEX,  
            1, (255, 255, 255), 2, cv2.LINE_AA
        )
        # cv2.putText(
        #     image, "State={}".format(self.state), (10, 80), 
        #     cv2.FONT_HERSHEY_SIMPLEX,  
        #     1, (255, 255, 255), 2, cv2.LINE_AA
        # )

        if self.goal_coords:
            x, y, x2, y2 = self.goal_coords
            cv2.rectangle(
                image, (x, y), (x2, y2), 
                (255, 0, 255), 3
            )
        for name, obj in self.objects.items():
                x, y = obj.x + obj.w // 2, obj.y - obj.h // 2
                cv2.putText(
                    image, name, (x-20, y-20), 
                    cv2.FONT_HERSHEY_SIMPLEX,  
                    1, (255, 255, 255), 2, cv2.LINE_AA
                )
                x, y = obj.x - obj.w // 2, obj.y - obj.h // 2
                cv2.rectangle(
                    image, (x, y), (x+obj.w, y+obj.h), 
                    (255, 255, 255), 4
                )
                if name == 'ball':
                    image[:, obj.x] = (255, 255, 255)
                    image[obj.y, :] = (255, 255, 255)
                # x, y = obj.x, obj.y
                # cv2.rectangle(
                #     image, (x-2, y-2), (x+2, y+2), 
                #     (255, 255, 255), -1
                # )
                # obj = obj.get_future()
                # x, y = obj.x - obj.w // 2, obj.y - obj.h // 2
                # cv2.rectangle(
                #     image, (x, y), (x+obj.w, y+obj.h), 
                #     (127, 127, 127), 3
                # )
                # x, y = obj.x, obj.y
                # cv2.rectangle(
                #     image, (x-2, y-2), (x+2, y+2), 
                #     (127, 127, 127), -1
                # )

        for obj in self.unmatched_objects:
            x, y = obj.x - obj.w // 2, obj.y - obj.h // 2
            label = obj.labels[-1]
            if label == 'ball':
                continue
                color = (255, 0, 0)
            elif label == 'referee':
                color = (0, 255, 255)
            elif label == 'player':
                color = (0, 255, 0)
            else:
                print(label)

            cv2.rectangle(
                image, (x, y), (x+obj.w, y+obj.h), 
                color, 2
            )
            # cv2.putText(
            #     image, '{:.2f}'.format(score), (x, y), 
            #     cv2.FONT_HERSHEY_SIMPLEX,  
            #     1, (255, 0, 0), 2, cv2.LINE_AA
            # )