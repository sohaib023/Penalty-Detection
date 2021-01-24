import os
import argparse

import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from libs.MultiTaskModel import MultiTaskModel
from libs.dataset import PenaltyDataset

from torch_references.engine import train_one_epoch, evaluate

def seg_to_image(seg):
    seg = torch.argmax(seg, dim=0).cpu().detach().numpy()
    image = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    image[seg == 1] = [0, 255, 0]
    image[seg == 2] = [255, 255, 255]
    return image
    
def collate_fn(batch):
    return tuple(zip(*batch))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-i',
        '--input_data_path',
        type=str,
        help="Path to directory containing input dataset.",
        default="data/detection"
    )
    '''
    {DATA_PATH}
        |-labels
        |-images
        |-val_images
    '''
    parser.add_argument(
        '-o',
        '--out_path',
        type=str,
        help="Path for outputting model weights and tensorboard summary.",
        default="output"
    )
    parser.add_argument(
        '-lr',
        '--learning_rate',
        type=float,
        help="Learning Rate",
        default=1e-4
    )
    parser.add_argument(
        '-e',
        '--epochs',
        type=int,
        help="Number of epochs to train",
        default=200
    )

    args = parser.parse_args()

    os.makedirs(args.out_path, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_dataset = PenaltyDataset(
        os.path.join(args.input_data_path, "images"), 
        os.path.join(args.input_data_path, "labels"),
        device=device, 
        transform=True
    )
    val_dataset = PenaltyDataset(
        os.path.join(args.input_data_path, "val_images"), 
        os.path.join(args.input_data_path, "labels"),
        device=device, 
        transform=False
    )

    # indices          = torch.randperm(len(dataset)).tolist()
    # val_split        = int(args.validation_split * len(indices))

    # train_dataset    = torch.utils.data.Subset(dataset, indices[val_split:])
    # val_dataset      = torch.utils.data.Subset(dataset, indices[:val_split])
    
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn, drop_last=True)
    val_dataloader   = DataLoader(val_dataset, batch_size=3, shuffle=True, collate_fn=collate_fn)

    model = MultiTaskModel(training=True)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    writer = SummaryWriter(os.path.join(args.out_path, "summary"))

    for epoch in range(args.epochs):
        model.train()

        train_one_epoch(model, optimizer, train_dataloader, device, epoch, writer, print_freq=10)
        coco_eval, seg_accuracy = evaluate(model, val_dataloader, device)
        ap = coco_eval.coco_eval['bbox'].stats[:6]

        for i, val in enumerate(ap):
            writer.add_scalar("coco_{}".format(i), val, epoch)
        writer.add_scalar("Val Seg", seg_accuracy, epoch)
        print("Validation Segmentation Accuracy: ", seg_accuracy)

        if (epoch + 1) % 5 == 0:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()
                },
                os.path.join(args.out_path, "epoch_{}.pth".format(epoch + 1))
            )