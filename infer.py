import os
import glob
import argparse

import cv2
import numpy as np
from PIL import Image

import torch
import torchvision.transforms.functional as TF

from libs.MultiTaskModel import MultiTaskModel

def seg_to_image(seg):
    seg = torch.argmax(seg, dim=0).cpu().detach().numpy()
    image = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    image[seg == 0] += np.array([0, 0, 255], dtype=np.uint8)[None, :]
    image[seg == 1] += np.array([0, 255, 0], dtype=np.uint8)[None, :]
    image[seg == 2] += np.array([255, 0, 0], dtype=np.uint8)[None, :]
    return image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-i',
        '--input_images',
        type=str,
        help="Path to directory containing input dataset.",
        default="data/detection/val_images"
    )
    parser.add_argument(
        '-o',
        '--out_path',
        type=str,
        help="Path for outputting model weights and tensorboard summary.",
        default="output/inference/"
    )
    parser.add_argument(
        '-c',
        '--checkpoint',
        type=str,
        help="Path to checkpoint file to be used for inference.",
        # default="output/inference/"
    )

    args = parser.parse_args()

    os.makedirs(args.out_path, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = MultiTaskModel(training=False)
    model.to(device)
    model.eval()

    ckpt = torch.load(args.checkpoint)
    model.load_state_dict(ckpt['model_state_dict'])


    INPUT_SHAPE = [750, 1333]
    CLASSES = ['player', 'referee', 'ball']

    for filename in glob.glob(os.path.join(args.input_images, "*.png")):
        print(filename)
        pil_image = Image.open(filename).convert("RGB")
        filename = os.path.basename(filename)

        org_w, org_h = pil_image.size[:2]
        
        image = TF.resize(pil_image, INPUT_SHAPE, Image.BILINEAR)
        image = TF.to_tensor(image).to(device)
        
        output_dict = model([image])[0]
        
        h, w = image.shape[-2:]
        rw = org_w/w
        rh = org_h/h

        # PROCESSING DETECTIONS

        image = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)

        for box, score, label in zip(output_dict['boxes'], output_dict['scores'], output_dict['labels']):
            if label != CLASSES.index('ball'):
                if score < 0.2:
                    continue
            elif score < 0.5:
                continue

            x, y, x2, y2 = box
            cv2.rectangle(image, (int(x * rw), int(y * rh)), (int(x2 * rw), int(y2 * rh)), (0, 255, 0), 2)
            cv2.putText(
                image, '{:.2f}'.format(score.item()), (int(x * rw), int(y * rh)), 
                cv2.FONT_HERSHEY_SIMPLEX,  
                1, (255, 0, 0), 2, cv2.LINE_AA
            ) 
            cv2.putText(
                image, CLASSES[label], (int(x * rw), int(y2 * rh) +20), 
                cv2.FONT_HERSHEY_SIMPLEX,  
                1, (255, 0, 0), 2, cv2.LINE_AA
            ) 
        cv2.imwrite(os.path.join(args.out_path, filename.replace('.png', '_det.png')), image)

        # PROCESSING SEGMENTATION MASK

        seg = output_dict['seg']
        seg = torch.nn.functional.interpolate(seg[None], size=[org_h, org_w], mode='nearest')[0]
        seg_image = seg_to_image(seg)
        cv2.imwrite(os.path.join(args.out_path, filename.replace('.png', '_seg.png')), seg_image)