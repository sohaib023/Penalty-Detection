import torch
import numpy as np

def seg_to_image(seg):
    seg = torch.argmax(seg, dim=0).cpu().detach().numpy()
    image = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    image[seg == 1] = [0, 255, 0]
    image[seg == 2] = [255, 255, 255]
    return image