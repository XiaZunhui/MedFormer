import sys, os
sys.path.append(os.path.dirname(sys.path[0]))
import torch
import torch.nn as nn
from timm.models.layers import LayerNorm2d
from models.medformer import MedFormer

class MedFormer_mm(MedFormer):
    def __init__(self, pretrained=None, **kwargs):
        super().__init__(**kwargs)
        
        # step 1: remove unused segmentation head & norm
        del self.head # classification head
        del self.norm # head norm

        # step 2: add extra norms for dense tasks
        self.extra_norms = nn.ModuleList()
        for i in range(4):
            self.extra_norms.append(LayerNorm2d(self.embed_dim[i]))
    
    def forward_features(self, x: torch.Tensor):
        out = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            out.append(self.extra_norms[i](x))
        return tuple(out)
    
    def forward(self, x:torch.Tensor):
        return self.forward_features(x)
