import torch
import torch.nn as nn
import torch.nn.functional as F

class DummyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = None

    def set_out_features(self, new_features):
        self.features = new_features

    def forward(self, _):
        return self.features