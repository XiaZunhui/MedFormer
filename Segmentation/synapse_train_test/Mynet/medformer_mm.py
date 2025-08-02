import sys, os

sys.path.append(os.path.dirname(sys.path[0]))
import torch
import torch.nn as nn

# from semantic_segmentation.mmcv_custom import load_checkpoint

# from mmseg.utils import get_root_logger
# from models_cls.biformer import BiFormer
from timm.models.layers import LayerNorm2d
from models.medformer import MedFormer
# from .Other_attention.swin.SwinFormer import MedFormer
# from .Other_attention.explict.explict_sparse import MedFormer
# from .Other_attention.dat.dat import MedFormer
# from .Other_attention.bra.medformer import MedFormer

class MedFormer_mm(MedFormer):
    def __init__(self, pretrained=None, **kwargs):
        super().__init__(**kwargs)

        # step 1: remove unused segmentation head & norm
        del self.head  # classification head
        del self.norm  # head norm

        # step 2: add extra norms for dense tasks
        self.extra_norms = nn.ModuleList()
        for i in range(4):
            self.extra_norms.append(LayerNorm2d(self.embed_dim[i]))

        # step 3: initialization & load ckpt
        # self.apply(self._init_weights)
        # self.init_weights(pretrained=pretrained)

        # 解决多卡环境下的同步问题，确保在多个设备上训练时批量归一化的准确性和一致性。
        # step 4: convert sync bn, as the batch size is too small in segmentation
        # TODO: check if this is correct
        # nn.SyncBatchNorm.convert_sync_batchnorm(self)

    # def init_weights(self, pretrained):
    #     if isinstance(pretrained, str):
    #         logger = get_root_logger()
    #         load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)
    #         print(f'Load pretrained model from {pretrained}')

    def forward_features(self, x: torch.Tensor):
        out = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            # DONE: check the inconsistency -> no effect on performance
            # in the version before submission:
            # x = self.extra_norms[i](x)
            # out.append(x)
            out.append(self.extra_norms[i](x))
        return tuple(out)

    def forward(self, x: torch.Tensor):
        return self.forward_features(x)
